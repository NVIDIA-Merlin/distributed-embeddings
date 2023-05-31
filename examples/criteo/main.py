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
"""Benchmarks of synthetic models"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from distributed_embeddings import IntegerLookup


def create_criteo_dataset():
  data = pd.read_csv(os.path.join(sys.path[0], 'train.txt'), delimiter='\t', header=None)
  sparse_features = list(range(14, 40))
  dense_features = list(range(1, 14))

  # load data in and fill 0 for missing ones
  data[sparse_features] = data[sparse_features].fillna('0',)
  data[dense_features] = data[dense_features].fillna(0.,).astype('float32')
  data[0] = data[0].fillna(0,).astype('int32')

  mms = MinMaxScaler(feature_range=(0, 1))
  data[dense_features] = mms.fit_transform(data[dense_features])

  # pandas store object(string) dtype, convert to int64 to use with integer lookup
  for feat_name in sparse_features:
    data[feat_name] = data[feat_name].apply(int, base=16)
  cat_feat = [data[[feat_name]] for feat_name in sparse_features]
  # we need zip to create list of tensor as cat input
  cat_datasets = [tf.data.Dataset.from_tensor_slices(feat) for feat in cat_feat]
  cat_dataset = tf.data.Dataset.zip(tuple(cat_datasets))

  # create numerical feature and lable dataset
  num_dataset = tf.data.Dataset.from_tensor_slices(data[dense_features])
  label_dataset = tf.data.Dataset.from_tensor_slices(data[[0]])

  # zip dataset together into expected structure
  train_dataset = tf.data.Dataset.zip(((num_dataset, cat_dataset), label_dataset))
  return train_dataset


class EmbeddingModel(tf.keras.Model):
  """A simple model for test"""

  def __init__(self, table_sizes):
    super().__init__()
    self.embeddings = []
    self.interger_lookups = []
    for size in table_sizes:
      self.embeddings.append(tf.keras.layers.Embedding(*size))
      self.interger_lookups.append(IntegerLookup(size[0]))
    self.mlp = tf.keras.Sequential()
    for size in [512, 256, 128]:
      self.mlp.add(tf.keras.layers.Dense(size, activation="relu"))
    self.mlp.add(tf.keras.layers.Dense(1, activation=None))

  def call(self, inputs):
    numerical_features, cat_features = inputs
    cat_features = [lookup(inp) for inp, lookup in zip(cat_features, self.interger_lookups)]
    outs = [tf.squeeze(emb(inp), 1) for inp, emb in zip(cat_features, self.embeddings)]
    outs = tf.concat(outs + [numerical_features], 1)
    outs = self.mlp(outs)
    return outs


def main():

  model = EmbeddingModel(26 * [[100000, 128]])
  optimizer = tf.keras.optimizers.SGD(learning_rate=0.03, momentum=0)
  bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                           from_logits=True)

  criteo_dataset = create_criteo_dataset()

  # TODO(deyuf): enable jit compile
  model.compile(optimizer=optimizer, loss=bce)
  model.fit(criteo_dataset.batch(1024), epochs=2)


if __name__ == '__main__':
  main()
