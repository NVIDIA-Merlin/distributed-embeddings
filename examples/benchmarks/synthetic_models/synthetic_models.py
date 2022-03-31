# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synthetic models for benchmark"""

import numpy as np

from absl import logging

import tensorflow as tf
from tensorflow import keras

import horovod.tensorflow as hvd

from distributed_embeddings.python.layers.embedding import Embedding
from distributed_embeddings.python.layers import dist_model_parallel as dmp


# pylint: disable=missing-type-doc
def power_law(k_min, k_max, alpha, r):
  """convert uniform distribution to power law distribution"""
  gamma = 1 - alpha
  y = pow(r * (pow(k_max, gamma) - pow(k_min, gamma)) + pow(k_min, gamma), 1.0 / gamma)
  return y.astype(np.int64)


def gen_power_law_data(batch_size, hotness, num_rows, alpha):
  """naive power law distribution generator

  NOTE: Repetition is allowed in multi hot data.
  TODO(skyw): Add support to disallow repetition
  """
  y = power_law(1, num_rows + 1, alpha, np.random.rand(batch_size * hotness)) - 1
  return tf.convert_to_tensor(y.reshape([batch_size, hotness]))


# pylint: enable=missing-type-doc


class InputGenerator():
  """Synthetic input generator

  Args:
    model_config (ModelConfig): A named tuple describes the synthetic model
    global_batch_size (int): Batch size.
    alpha (float): exponent to generate power law distributed input. 0 means uniform, default 0
    input_ids_list (list of int): Nested list containing input indices by rank.
    num_batches (int): Number of batches to generate. Default 100.
    dp_input (bool): If True, generate data parallel input. Default False.
  """

  def __init__(self,
               model_config,
               global_batch_size,
               alpha=0,
               input_ids_list=None,
               num_batches=10,
               dp_input=False):
    self.global_batch_size = global_batch_size
    self.dp_batch_size = global_batch_size // hvd.size()
    self.num_batches = num_batches
    self.dp_input = dp_input

    input_count = 0
    embed_count = 0
    global_input_shapes = []
    for config in model_config.embedding_configs:
      for _ in range(config.num_tables):
        for hotness in config.nnz:
          global_input_shapes.append([hotness, config.num_rows])
          input_count += 1
        embed_count += 1
    logging.info("Generated %d categorical inputs for %d embedding tables", input_count,
                 embed_count)

    self.input_pool = []
    for _ in range(num_batches):
      cat_features = []
      for input_id in input_ids_list[hvd.rank()]:
        hotness, num_rows = global_input_shapes[input_id]
        if alpha == 0:
          cat_features.append(
              tf.random.uniform(shape=[global_batch_size, hotness], maxval=num_rows,
                                dtype=tf.int64))
        else:
          cat_features.append(gen_power_law_data(global_batch_size, hotness, num_rows, alpha))

      numerical_features = tf.random.uniform(
          shape=[self.dp_batch_size, model_config.num_numerical_features],
          maxval=100,
          dtype=tf.float32)
      labels = tf.random.uniform(shape=[self.dp_batch_size, 1], maxval=2, dtype=tf.int32)

      self.input_pool.append((numerical_features, cat_features, labels))

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return self.input_pool[idx]


class SyntheticModel(keras.Model):  # pylint: disable=abstract-method
  """Main synthetic model class

  Args:
    model_config (ModelConfig): A named tuple describes the synthetic model
    column_slice_threshold (int or None): upper bound of elements count in each slice
    dp_input (bool): If True, use data parallel input. Otherwise model parallel input.
        Default False.
  """

  def __init__(self, model_config, column_slice_threshold=None, dp_input=False):
    if dp_input:
      raise NotImplementedError

    super().__init__()
    self.num_numerical_features = model_config.num_numerical_features

    # Expand embedding configs and create embeddings
    embedding_layers = []
    self.input_table_map = []
    embed_count = 0
    for config in model_config.embedding_configs:
      if len(config.nnz) > 1 and not config.shared:
        raise NotImplementedError("Nonshared multihot embedding is not implemented yet")

      for _ in range(config.num_tables):
        embedding_layers.append(Embedding(config.num_rows, config.width, combiner='sum'))
        for _ in range(len(config.nnz)):
          self.input_table_map.append(embed_count)
        embed_count += 1
    logging.info("%d embedding tables created.", embed_count)
    self.embeddings = dmp.DistributedEmbedding(embedding_layers,
                                               strategy="memory_balanced",
                                               dp_input=dp_input,
                                               input_table_map=self.input_table_map,
                                               column_slice_threshold=column_slice_threshold)

    # Use a memory bandwidth limited pooling layer to emulate interaction (aka FM, pool, etc.)
    if model_config.interact_stride is not None:
      self.interact = keras.layers.AveragePooling1D(model_config.interact_stride,
                                                    padding='same',
                                                    data_format='channels_first')
    else:
      self.interact = None  # use concatenation

    # Create MLP
    self.mlp = keras.Sequential()
    for size in model_config.mlp_sizes:
      self.mlp.add(keras.layers.Dense(size, activation="relu"))
    self.mlp.add(keras.layers.Dense(1, activation=None))

    del embedding_layers

  def call(self, inputs):
    numerical_features, cat_features = inputs

    x = self.embeddings(cat_features)

    if self.interact is not None:
      x = [tf.squeeze(self.interact(tf.expand_dims(tf.concat(x, 1), axis=0)))]
    x = tf.concat(x + [numerical_features], 1)

    x = self.mlp(x)
    return x
