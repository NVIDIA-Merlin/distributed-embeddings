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
# pylint:disable=missing-docstring, no-self-use, invalid-name
import tensorflow as tf
from distributed_embeddings import embedding_lookup


class EmbeddingLookupTest(tf.test.TestCase):

  def test_variable_hotness(self):
    voc, emb, batch, max_hotness = 69, 64, 15, 207
    # create dense representation of index matrix
    data_a = tf.random.uniform(shape=[batch, max_hotness], minval=1, maxval=max_hotness + 1)
    data_b = tf.random.uniform(shape=[batch], minval=1, maxval=max_hotness + 1)
    # make sure there is no empty row
    data_c = tf.reshape(tf.eye(max_hotness, batch_shape=[batch // max_hotness + 1]),
                        [-1, max_hotness])[:batch]

    data_0 = tf.cast((data_a / tf.reshape(data_b, [-1, 1]) + data_c) > 1, tf.int64)
    data_1 = tf.random.uniform(shape=[batch, max_hotness], minval=0, maxval=voc, dtype=tf.int64)
    data = data_0 * data_1

    # COO format for tf native API
    ref_ids = tf.sparse.from_dense(data)
    # Ragged format. We can't use ragged.from_sparse() since it is not ragged-right, see:
    # https://github.com/tensorflow/tensorflow/blob/v2.5.0/tensorflow/python/ops/ragged/ragged_tensor.py#L2678
    row_lengths = tf.math.count_nonzero(data, axis=1)
    ids = tf.RaggedTensor.from_row_lengths(ref_ids.values, row_lengths)

    initial_weight = tf.random.uniform([voc, emb], dtype=tf.float32)
    param = tf.Variable(initial_weight)

    for red in ['sum', 'mean']:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(param)
        ref_ret = tf.nn.embedding_lookup_sparse(param, ref_ids, sp_weights=None, combiner=red)
        ret = embedding_lookup(param, ids, combiner=red)
      ref_g = tape.gradient(ref_ret, param)
      g = tape.gradient(ret, param)

      ref_g_dense = tf.convert_to_tensor(ref_g)
      g_dense = tf.convert_to_tensor(g)
      # Seems some ops in sparse lookup is running on CPU and rounding differently
      self.assertAllClose(ref_ret, ret)
      self.assertAllClose(ref_g_dense, g_dense)

  def test_constant_hotness(self):
    voc, emb, batch, hotness = 69, 64, 15, 207
    ids = tf.random.uniform(shape=[batch, hotness], minval=0, maxval=voc, dtype=tf.int64)

    initial_weight = tf.random.uniform([voc, emb], dtype=tf.float32)
    param = tf.Variable(initial_weight)

    for red in ['sum', 'mean']:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(param)
        if red == 'sum':
          ref_ret = tf.reduce_sum(tf.nn.embedding_lookup(param, ids), 1)
        if red == 'mean':
          ref_ret = tf.reduce_mean(tf.nn.embedding_lookup(param, ids), 1)
        ret = embedding_lookup(param, ids, combiner=red)
      ref_g = tape.gradient(ref_ret, param)
      g = tape.gradient(ret, param)

      ref_g_dense = tf.convert_to_tensor(ref_g)
      g_dense = tf.convert_to_tensor(g)

      self.assertAllEqual(ref_ret, ret)
      self.assertAllEqual(ref_g_dense, g_dense)

  def test_sparse_tensor_input(self):
    voc, emb, batch, max_hotness = 69, 64, 15, 207
    # create dense representation of index matrix
    data_a = tf.random.uniform(shape=[batch, max_hotness], minval=1, maxval=max_hotness + 1)
    data_b = tf.random.uniform(shape=[batch], minval=1, maxval=max_hotness + 1)
    # make sure there is no empty row
    data_c = tf.reshape(tf.eye(max_hotness, batch_shape=[batch // max_hotness + 1]),
                        [-1, max_hotness])[:batch]

    data_0 = tf.cast((data_a / tf.reshape(data_b, [-1, 1]) + data_c) > 1, tf.int64)
    data_1 = tf.random.uniform(shape=[batch, max_hotness], minval=0, maxval=voc, dtype=tf.int64)
    data = data_0 * data_1

    # COO format for tf native API
    ref_ids = tf.sparse.from_dense(data)
    test_ids = tf.sparse.from_dense(data)

    initial_weight = tf.random.uniform([voc, emb], dtype=tf.float32)
    param = tf.Variable(initial_weight)

    for red in ['sum', 'mean']:
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(param)
        ref_ret = tf.nn.embedding_lookup_sparse(param, ref_ids, sp_weights=None, combiner=red)
        ret = embedding_lookup(param, test_ids, combiner=red)
      ref_g = tape.gradient(ref_ret, param)
      g = tape.gradient(ret, param)

      ref_g_dense = tf.convert_to_tensor(ref_g)
      g_dense = tf.convert_to_tensor(g)
      # Seems some ops in sparse lookup is running on CPU and rounding differently
      self.assertAllClose(ref_ret, ret)
      self.assertAllClose(ref_g_dense, g_dense)


if __name__ == '__main__':
  tf.test.main()
