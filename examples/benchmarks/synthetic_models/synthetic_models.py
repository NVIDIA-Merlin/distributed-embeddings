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

import distributed_embeddings
from distributed_embeddings import dist_model_parallel as dmp

tf.random.set_seed(12345)
rng = np.random.default_rng(12345)
np.random.seed(12345)

# pylint: disable=missing-type-doc
def power_law(k_min, k_max, alpha, r, dtype='int64'):
  """convert uniform distribution to power law distribution"""
  gamma = 1 - alpha
  y = pow(r * (pow(k_max, gamma) - pow(k_min, gamma)) + pow(k_min, gamma), 1.0 / gamma)
  if dtype == 'int32':
    return y.astype(np.int32)
  return y.astype(np.int64)


def gen_power_law_data(batch_size, hotness, num_rows, alpha, dtype):
  """naive power law distribution generator

  NOTE: Repetition is allowed in multi hot data.
  TODO(skyw): Add support to disallow repetition
  """
  y = power_law(1, num_rows + 1, alpha, np.random.rand(batch_size * hotness), dtype) - 1
  return tf.convert_to_tensor(y.reshape([batch_size, hotness]))


# pylint: enable=missing-type-doc
def generate_cat_input(batch_size, max_hotness, min_hotness, mean_hotness, num_rows, alpha, dtype):
  if dtype == 'int32':
    tfdtype = tf.int32
  else:
    tfdtype = tf.int64

  if max_hotness == min_hotness:
    if alpha == 0:
      return max_hotness * batch_size, tf.random.uniform(shape=[max_hotness, batch_size], maxval=num_rows, dtype=tfdtype)
    return max_hotness * batch_size, gen_power_law_data(batch_size, max_hotness, num_rows, alpha, dtype)

  n = max_hotness - min_hotness
  if n == 0:
    batch_lengths = [min_hotness for _ in range(batch_size)]
  else:
    p = (mean_hotness - min_hotness) / n
    batch_lengths = rng.binomial(n, p, batch_size) + min_hotness
  total_elems = np.sum(batch_lengths)
  # Do not generate an empty batch.
  if total_elems == 0:
    batch_lengths[0] = 1
    total_elems += 1
  if alpha == 0:
    values = tf.random.uniform(shape=[total_elems],
                                    maxval=num_rows,
                                    dtype=tfdtype)
  else:
    values = power_law(1, num_rows + 1, alpha, np.random.rand(total_elems), dtype) - 1
  batch_lengths = np.insert(batch_lengths, 0, 0)
  row_splits = np.cumsum(batch_lengths)
  return total_elems, tf.RaggedTensor.from_row_splits(values=values, row_splits=row_splits, validate=False).with_row_splits_dtype(tfdtype)

# pylint: enable=missing-type-doc


class InputGenerator(keras.utils.Sequence):
  """Synthetic input generator

  Args:
    model_config (ModelConfig): A named tuple describes the synthetic model
    global_batch_size (int): Batch size.
    alpha (float): exponent to generate power law distributed input. 0 means uniform, default 0
    mp_input_ids (list of int): List containing model parallel input indices.
    num_batches (int): Number of batches to generate. Default 100.
    embedding_device (string): device to put embedding and inputs on
  """

  def __init__(self,
               model_config,
               global_batch_size,
               alpha=0,
               mp_input_ids=None,
               num_batches=10,
               embedding_device='/GPU:0',
               mean_hotness_ratio=1.0):
    self.dp_batch_size = global_batch_size // hvd.size()
    self.cat_batch_size = global_batch_size if mp_input_ids is not None else self.dp_batch_size
    self.num_batches = num_batches
    self.dtype = 'int64'

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
    total_cat_elems_generated = 0
    for _ in range(num_batches):
      cat_features = []
      input_ids = mp_input_ids if mp_input_ids is not None else list(range(input_count))
      for input_id in input_ids:
        hotness, num_rows = global_input_shapes[input_id]
        with tf.device(embedding_device):
          max_hotness = hotness
          if mean_hotness_ratio < 1:
            #min_hotness = 0
            #mean_hotness = hotness * mean_hotness_ratio
            min_hotness = 1
            mean_hotness = min_hotness + hotness * mean_hotness_ratio
          else:
            min_hotness = hotness
            mean_hotness = hotness
          cat_elems_generated, data = generate_cat_input(self.cat_batch_size, max_hotness, min_hotness, mean_hotness, num_rows, alpha, self.dtype)
          total_cat_elems_generated += cat_elems_generated
          cat_features.append(data)
          numerical_features = tf.random.uniform(
              shape=[self.dp_batch_size, model_config.num_numerical_features],
              maxval=100,
              dtype=tf.float32)
          labels = tf.random.uniform(shape=[self.dp_batch_size, 1], maxval=2, dtype=tf.int32)

      self.input_pool.append(((numerical_features, cat_features), labels))
    logging.info('Generated %d categorical lookup ids for %d batches. Avg of %f ids per sample', total_cat_elems_generated, num_batches, total_cat_elems_generated / num_batches / self.cat_batch_size) 

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return self.input_pool[idx]


class SyntheticModelTFDE(keras.Model):  # pylint: disable=abstract-method
  """Main synthetic model class

  Args:
    model_config (ModelConfig): A named tuple describes the synthetic model
    column_slice_threshold (int or None): upper bound of elements count in each slice
    dp_input (bool): If True, use data parallel input. Otherwise model parallel input.
        Default False.
  """

  def __init__(self, model_config, column_slice_threshold=None, dp_input=False):
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
        embedding_layers.append(
            distributed_embeddings.Embedding(config.num_rows, config.width, combiner='sum'))
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


class SyntheticModelNative(keras.Model):  # pylint: disable=abstract-method
  """Main synthetic model class

  Args:
    model_config (ModelConfig): A named tuple describes the synthetic model
    column_slice_threshold (int or None): upper bound of elements count in each slice
    dp_input (bool): If True, use data parallel input. Otherwise model parallel input.
        Default False.
  """

  def __init__(self, model_config, embedding_device='/GPU:0'):
    super().__init__()
    self.num_numerical_features = model_config.num_numerical_features
    self.embedding_device = embedding_device
    # Expand embedding configs and create embeddings
    self.embeddings = []
    self.input_table_map = []
    embed_count = 0
    for config in model_config.embedding_configs:
      if len(config.nnz) > 1 and not config.shared:
        raise NotImplementedError("Nonshared multihot embedding is not implemented yet")

      for _ in range(config.num_tables):
        self.embeddings.append(tf.keras.layers.Embedding(config.num_rows, config.width))
        for _ in range(len(config.nnz)):
          self.input_table_map.append(embed_count)
        embed_count += 1
    logging.info("%d embedding tables created.", embed_count)

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

  def call(self, inputs):
    numerical_features, cat_features = inputs

    with tf.device(self.embedding_device):
      x = [self.embeddings[ind](inp) for ind, inp in zip(self.input_table_map, cat_features)]
    x = [tf.reduce_sum(t, 1) for t in x]

    if self.interact is not None:
      x = [tf.squeeze(self.interact(tf.expand_dims(tf.concat(x, 1), axis=0)))]
    x = tf.concat(x + [numerical_features], 1)

    x = self.mlp(x)
    return x
