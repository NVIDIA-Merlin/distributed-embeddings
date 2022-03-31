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
"""DLRM Sample using model parallel embedding"""

import os
import concurrent
import math
import queue
from typing import Optional, Sequence, Tuple
import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class DLRMInitializer(tf.keras.initializers.Initializer):
  """ dlrm embedding weight initializer
  """

  def __call__(self, shape, dtype=tf.float32):
    maxval = tf.sqrt(tf.constant(1.) / tf.cast(shape[0], tf.float32))
    maxval = tf.cast(maxval, dtype=dtype)
    minval = -maxval

    weights = tf.random.uniform(shape, minval=minval, maxval=maxval, dtype=dtype)
    weights = tf.cast(weights, dtype=tf.float32)
    return weights

  def get_config(self):
    return {}


# pylint: disable=line-too-long
class LearningRateScheduler:
  """
  LR Scheduler combining Polynomial Decay with Warmup at the beginning.
  TF-based cond operations necessary for performance in graph mode.
  """

  def __init__(self, optimizers, base_lr, warmup_steps, decay_start_step, decay_steps):
    self.optimizers = optimizers
    self.warmup_steps = tf.constant(warmup_steps, dtype=tf.int32)
    self.decay_start_step = tf.constant(decay_start_step, dtype=tf.int32)
    self.decay_steps = tf.constant(decay_steps)
    self.decay_end_step = decay_start_step + decay_steps
    self.poly_power = 2
    self.base_lr = base_lr
    with tf.device('/CPU:0'):
      self.step = tf.Variable(0)

  @tf.function
  def __call__(self):
    with tf.device('/CPU:0'):
      # used for the warmup stage
      warmup_step = tf.cast(1 / self.warmup_steps, tf.float32)
      lr_factor_warmup = 1 - tf.cast(self.warmup_steps - self.step, tf.float32) * warmup_step
      lr_factor_warmup = tf.cast(lr_factor_warmup, tf.float32)

      # used for the constant stage
      lr_factor_constant = tf.cast(1., tf.float32)

      # used for the decay stage
      lr_factor_decay = (self.decay_end_step - self.step) / self.decay_steps
      lr_factor_decay = tf.math.pow(lr_factor_decay, self.poly_power)
      lr_factor_decay = tf.cast(lr_factor_decay, tf.float32)

      poly_schedule = tf.cond(self.step < self.decay_start_step, lambda: lr_factor_constant,
                              lambda: lr_factor_decay)

      lr_factor = tf.cond(self.step < self.warmup_steps, lambda: lr_factor_warmup,
                          lambda: poly_schedule)

      lr = self.base_lr * lr_factor
      for optimizer in self.optimizers:
        optimizer.lr.assign(lr)

      self.step.assign(self.step + 1)


# dot interact but with concating mlp inside for simpler model building
def dot_interact(emb_outs, bottom_mlp_out):
  concat_features = tf.concat([bottom_mlp_out] + emb_outs, axis=1)
  concat_features = tf.reshape(concat_features, [-1, len(emb_outs) + 1, bottom_mlp_out.shape[-1]])

  # Interact features, select lower-triangular portion, and re-shape.
  interactions = tf.matmul(concat_features, concat_features, transpose_b=True)

  ones = tf.ones_like(interactions, dtype=tf.float32)
  upper_tri_mask = tf.linalg.band_part(ones, 0, -1)

  feature_dim = tf.shape(interactions)[-1]

  lower_tri_mask = ones - upper_tri_mask
  activations = tf.boolean_mask(interactions, lower_tri_mask)
  out_dim = feature_dim * (feature_dim - 1) // 2

  activations = tf.reshape(activations, shape=[-1, out_dim])

  # concat mlp out again
  activations = tf.concat([activations, bottom_mlp_out], axis=1)

  return activations


def get_categorical_feature_type(size: int):
  types = (np.int8, np.int16, np.int32)

  for numpy_type in types:
    if size < np.iinfo(numpy_type).max:
      return numpy_type

  raise RuntimeError(f"Categorical feature of size {size} is too big for defined types")


class DummyDataset:
  """Dummy dataset used for benchmarking
  """

  def __init__(self, flags, num_workers, num_table, is_train, dp_input):
    # create mp inputs for embeddings and dp inputs for rest
    self.numerical_features = tf.zeros(
        shape=[flags.batch_size // num_workers, flags.num_numerical_features])
    if dp_input:
      self.categorical_features = [
          tf.zeros(shape=[flags.batch_size // num_workers], dtype=tf.int64)
      ] * num_table
    else:
      self.categorical_features = [tf.zeros(shape=[flags.batch_size], dtype=tf.int64)] * num_table

    if is_train:
      self.labels = tf.ones(shape=[flags.batch_size // num_workers, 1])
    else:
      self.labels = tf.ones(shape=[flags.batch_size, 1])
    self.num_batches = flags.num_batches

  def __getitem__(self, idx):
    if idx >= self.num_batches:
      raise StopIteration

    return self.numerical_features, self.categorical_features, self.labels

  def __len__(self):
    return self.num_batches


class RawBinaryDataset:
  """Split version of Criteo dataset
    Args:
      data_path (str): Full path to split binary file of dataset. It must contain numerical.bin, label.bin and
          cat_0 ~ cat_25.bin
      batch_size (int):
      numerical_features(boolean): Number of numerical features to load, default=0 (don't load any)
      categorical_features (list or None): categorical features used by the rank (IDs of the features)
      categorical_feature_sizes (list of integers): max value of each of the categorical features
      prefetch_depth (int): How many samples to prefetch. Default 10.
  """

  def __init__(
      self,
      data_path: str,
      batch_size: int = 1,
      numerical_features: int = 0,
      categorical_features: Optional[Sequence[int]] = None,
      categorical_feature_sizes: Optional[Sequence[int]] = None,
      prefetch_depth: int = 10,
      drop_last_batch: bool = False,
      valid: bool = False,
      offset: int = -1,
      lbs: int = -1,
      dp_input: bool = False,
  ):
    suffix = 'test' if valid else 'train'
    data_path = os.path.join(data_path, suffix)
    self._label_bytes_per_batch = np.dtype(np.bool).itemsize * batch_size
    self._numerical_bytes_per_batch = numerical_features * np.dtype(
        np.float16).itemsize * batch_size
    self._numerical_features = numerical_features

    self._categorical_feature_types = [
        get_categorical_feature_type(size) for size in categorical_feature_sizes
    ] if categorical_feature_sizes else []
    self._categorical_bytes_per_batch = [
        np.dtype(cat_type).itemsize * batch_size for cat_type in self._categorical_feature_types
    ]
    self._categorical_features = categorical_features
    self._batch_size = batch_size
    self._label_file = os.open(os.path.join(data_path, 'label.bin'), os.O_RDONLY)
    self._num_entries = int(math.ceil(os.fstat(self._label_file).st_size
                                      / self._label_bytes_per_batch)) if not drop_last_batch \
                        else int(math.floor(os.fstat(self._label_file).st_size / self._label_bytes_per_batch))

    if numerical_features > 0:
      self._numerical_features_file = os.open(os.path.join(data_path, "numerical.bin"), os.O_RDONLY)
      number_of_numerical_batches = math.ceil(os.fstat(self._numerical_features_file).st_size
                                              / self._numerical_bytes_per_batch) if not drop_last_batch \
                                    else math.floor(os.fstat(self._numerical_features_file).st_size
                                                    / self._numerical_bytes_per_batch)
      if number_of_numerical_batches != self._num_entries:
        raise ValueError(
            f"Size mismatch in data files. Expected: {self._num_entries}, got: {number_of_numerical_batches}"
        )
    else:
      self._numerical_features_file = None

    if categorical_features:
      self._categorical_features_files = []
      for cat_id in categorical_features:
        cat_file = os.open(os.path.join(data_path, f"cat_{cat_id}.bin"), os.O_RDONLY)
        cat_bytes = self._categorical_bytes_per_batch[cat_id]
        number_of_categorical_batches = math.ceil(os.fstat(cat_file).st_size / cat_bytes) if not drop_last_batch \
                                        else math.floor(os.fstat(cat_file).st_size / cat_bytes)
        if number_of_categorical_batches != self._num_entries:
          raise ValueError(
              f"Size mismatch in data files. Expected: {self._num_entries}, got: {number_of_categorical_batches}"
          )
        self._categorical_features_files.append(cat_file)
    else:
      self._categorical_features_files = None

    self._prefetch_depth = min(prefetch_depth, self._num_entries)
    self._prefetch_queue = queue.Queue()
    self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    self.offset = offset
    self.lbs = lbs
    self.valid = valid
    self.dp_input = dp_input

  def __len__(self):
    return self._num_entries

  def __getitem__(self, idx: int):
    if idx >= self._num_entries:
      raise IndexError()

    if self._prefetch_depth <= 1:
      return self._get_item(idx)

    if idx == 0:
      for i in range(self._prefetch_depth):
        self._prefetch_queue.put(self._executor.submit(self._get_item, (i)))
    if idx < self._num_entries - self._prefetch_depth:
      self._prefetch_queue.put(self._executor.submit(self._get_item, (idx + self._prefetch_depth)))
    return self._prefetch_queue.get().result()

  def _get_item(self, idx: int) -> Tuple[tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
    click = self._get_label(idx)
    numerical_features = self._get_numerical_features(idx)
    categorical_features = self._get_categorical_features(idx)
    if self.offset >= 0:
      if not self.valid:
        click = click[self.offset:self.offset + self.lbs]
      numerical_features = numerical_features[self.offset:self.offset + self.lbs]
      if self.dp_input:
        categorical_features = [f[self.offset:self.offset + self.lbs] for f in categorical_features]
    return numerical_features, categorical_features, click

  def _get_label(self, idx: int) -> tf.Tensor:
    raw_label_data = os.pread(self._label_file, self._label_bytes_per_batch,
                              idx * self._label_bytes_per_batch)
    array = np.frombuffer(raw_label_data, dtype=np.bool)
    array = tf.convert_to_tensor(array, dtype=tf.float32)
    array = tf.expand_dims(array, 1)
    return array

  def _get_numerical_features(self, idx: int) -> Optional[tf.Tensor]:
    if self._numerical_features_file is None:
      return -1

    raw_numerical_data = os.pread(self._numerical_features_file, self._numerical_bytes_per_batch,
                                  idx * self._numerical_bytes_per_batch)
    array = np.frombuffer(raw_numerical_data, dtype=np.float16)
    array = tf.convert_to_tensor(array)
    return tf.reshape(array, shape=[self._batch_size, self._numerical_features])

  def _get_categorical_features(self, idx: int) -> Optional[tf.Tensor]:
    if self._categorical_features_files is None:
      return -1

    categorical_features = []
    for cat_id, cat_file in zip(self._categorical_features, self._categorical_features_files):
      cat_bytes = self._categorical_bytes_per_batch[cat_id]
      cat_type = self._categorical_feature_types[cat_id]
      raw_cat_data = os.pread(cat_file, cat_bytes, idx * cat_bytes)
      array = np.frombuffer(raw_cat_data, dtype=cat_type)
      tensor = tf.convert_to_tensor(array)
      categorical_features.append(tensor)
    return categorical_features

  def __del__(self):
    data_files = [self._label_file, self._numerical_features_file]
    if self._categorical_features_files is not None:
      data_files += self._categorical_features_files

    for data_file in data_files:
      if data_file is not None:
        os.close(data_file)
