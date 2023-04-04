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
"""Test of embedding layers"""

import tensorflow as tf
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.platform import test
from tensorflow.python.keras import keras_parameterized, testing_utils, combinations
from tensorflow.python.training import adagrad
from tensorflow.python.ops.ragged import ragged_factory_ops
from distributed_embeddings.python.layers import embedding


# pylint:disable=missing-docstring, no-self-use
class EmbeddingTest(keras_parameterized.TestCase):

  @keras_parameterized.run_all_keras_modes
  def test_1d_input(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3)
    model = tf.keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    model.run_eagerly = testing_utils.should_run_eagerly()
    outputs = model(tf.constant([0, 1, 0], dtype='int64'))
    self.assertAllEqual(outputs, [[1, 2], [3, 4], [1, 2]])

  @keras_parameterized.run_all_keras_modes
  def test_2d_input_no_combiner(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3)
    model = tf.keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    model.run_eagerly = testing_utils.should_run_eagerly()
    outputs = model.predict(np.array([[0, 1], [2, 0]], dtype='int64'))
    self.assertAllEqual(outputs, [[[1, 2], [3, 4]], [[5, 6], [1, 2]]])

  @keras_parameterized.run_all_keras_modes
  def test_2d_input_with_sum_combiner(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3, combiner='sum')
    model = tf.keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    model.run_eagerly = testing_utils.should_run_eagerly()
    outputs = model.predict(np.array([[0, 1], [2, 0]], dtype='int64'))
    self.assertAllEqual(outputs, [[4, 6], [6, 8]])

  @keras_parameterized.run_all_keras_modes
  def test_3d_input_no_combiner(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3)
    model = tf.keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    model.run_eagerly = testing_utils.should_run_eagerly()
    ids = np.array([[[0, 1], [2, 0], [1, 2]]], dtype='int64')
    outputs = model.predict(ids)
    self.assertAllEqual(outputs, [[[[1, 2], [3, 4]], [[5, 6], [1, 2]], [[3, 4], [5, 6]]]])

  @keras_parameterized.run_all_keras_modes
  def test_3d_input_with_mean_combiner(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3, combiner='mean')
    model = tf.keras.models.Sequential([layer])

    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    model.run_eagerly = testing_utils.should_run_eagerly()
    ids = np.array([[[0, 1], [2, 0], [1, 2]]], dtype='int64')
    outputs = model.predict(ids)
    self.assertAllEqual(outputs, [[[2, 3], [3, 4], [4, 5]]])

  @keras_parameterized.run_all_keras_modes
  def test_ragged_input(self):
    layer = embedding.Embedding(input_dim=3,
                                output_dim=2,
                                weights=[np.array([[0., 3.], [1., 5.], [7., 2.]])])
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, ragged=True)
    outputs = layer(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.run_eagerly = testing_utils.should_run_eagerly()
    ids = ragged_factory_ops.constant([[1, 2, 2], [0], [1, 2]], ragged_rank=1)
    outputs = model.predict(ids)

    ref_layer = tf.keras.layers.Embedding(input_dim=3,
                                          output_dim=2,
                                          weights=[np.array([[0., 3.], [1., 5.], [7., 2.]])])
    ref_outputs = ref_layer(ids)
    self.assertAllEqual(outputs, ref_outputs)

  @keras_parameterized.run_all_keras_modes
  def test_ragged_input_with_mean_combiner(self):
    layer = embedding.Embedding(input_dim=3,
                                output_dim=2,
                                combiner='mean',
                                weights=[np.array([[0., 3.], [1., 5.], [7., 2.]])])
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, ragged=True)
    outputs = layer(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.run_eagerly = testing_utils.should_run_eagerly()
    outputs = model.predict(ragged_factory_ops.constant([[1, 2, 2], [0], [1, 2]], ragged_rank=1))
    self.assertAllEqual(outputs, [[5., 3.], [0., 3.], [4., 3.5]])

  @keras_parameterized.run_all_keras_modes
  def test_sparse_input_with_mean_combiner(self):
    layer = embedding.Embedding(input_dim=3,
                                output_dim=2,
                                combiner='mean',
                                weights=[np.array([[0., 3.], [1., 5.], [7., 2.]])])
    inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.int64, sparse=True)
    outputs = layer(inputs)

    model = tf.keras.Model(inputs, outputs)
    model.run_eagerly = testing_utils.should_run_eagerly()

    outputs = model.predict(
        tf.sparse.SparseTensor(indices=[[0, 0], [0, 1], [0, 2], [1, 0], [2, 0], [2, 1]],
                               values=[1, 2, 2, 0, 1, 2],
                               dense_shape=[3, 4]))
    self.assertAllEqual(outputs, [[5., 3.], [0., 3.], [4., 3.5]])

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_2d_input_with_sum_combiner_with_grad(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3, combiner='sum')
    layer.build((None, 2))
    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    inputs = tf.keras.backend.constant([[0, 1, 0]], dtype='int64')
    with backprop.GradientTape() as tape:
      output = layer(inputs)
    gs = tape.gradient(output, layer.weights)
    opt = adagrad.AdagradOptimizer(0.1)
    opt.apply_gradients(zip(gs, layer.weights))

    ref_layer = tf.keras.layers.Embedding(output_dim=2, input_dim=3)
    ref_layer.build((None, 2))
    ref_layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    # grad of sum combiner is same as grads for flatten inputs without combiner
    ref_inputs = tf.keras.backend.constant([0, 1, 0], dtype='int64')
    with backprop.GradientTape() as tape:
      ref_output = ref_layer(ref_inputs)
    ref_gs = tape.gradient(ref_output, ref_layer.weights)
    ref_opt = adagrad.AdagradOptimizer(0.1)
    ref_opt.apply_gradients(zip(ref_gs, ref_layer.weights))
    self.assertAllEqual(layer.weights[0], ref_layer.weights[0])
    self.assertAllEqual(tf.convert_to_tensor(gs[0]), tf.convert_to_tensor(ref_gs[0]))

  @combinations.generate(combinations.combine(mode=['eager']))
  def test_2d_input_with_sum_combiner_with_grad_32bit(self):
    layer = embedding.Embedding(output_dim=2, input_dim=3, combiner='sum')
    layer.build((None, 2))
    layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    inputs = tf.keras.backend.constant([[0, 1, 0]], dtype='int32')
    with backprop.GradientTape() as tape:
      output = layer(inputs)
    gs = tape.gradient(output, layer.weights)
    opt = adagrad.AdagradOptimizer(0.1)
    opt.apply_gradients(zip(gs, layer.weights))

    ref_layer = tf.keras.layers.Embedding(output_dim=2, input_dim=3)
    ref_layer.build((None, 2))
    ref_layer.set_weights([np.array([[1, 2], [3, 4], [5, 6]])])
    # grad of sum combiner is same as grads for flatten inputs without combiner
    ref_inputs = tf.keras.backend.constant([0, 1, 0], dtype='int32')
    with backprop.GradientTape() as tape:
      ref_output = ref_layer(ref_inputs)
    ref_gs = tape.gradient(ref_output, ref_layer.weights)
    ref_opt = adagrad.AdagradOptimizer(0.1)
    ref_opt.apply_gradients(zip(ref_gs, ref_layer.weights))
    self.assertAllEqual(layer.weights[0], ref_layer.weights[0])
    self.assertAllEqual(tf.convert_to_tensor(gs[0]), tf.convert_to_tensor(ref_gs[0]))


class ConcatOneHotEmbeddingTest(test.TestCase):

  def test_smoke(self):
    feature_sizes = [3, 5, 7, 11]
    embedding_width = 3
    test_embedding = embedding.ConcatOneHotEmbedding(feature_sizes, embedding_width)
    indices = tf.constant([[1, 2, 3, 4], [2, 4, 6, 10]])
    test_embedding(indices)


if __name__ == "__main__":
  test.main()
