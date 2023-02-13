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
"""Embedding layers"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import constraints
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
from tensorflow.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.ops.ragged import ragged_tensor
from distributed_embeddings.python.ops import embedding_lookup_ops


class CPUInitializer(tf.keras.initializers.Initializer):
  """ initializer wrapper to force one-time init onto CPU, avoiding OOM
  """

  def __init__(self, initializer):
    self._initializer = initializer

  def __call__(self, shape, dtype=None, **kwargs):
    with tf.device('/CPU:0'):
      res = self._initializer(shape, **kwargs)
    return res


class Embedding(tf.keras.layers.Layer):
  """Turns indices into vectors of fixed size.

  Args:
    input_dim (int): Size of the vocabulary, i.e. maximum index + 1.
    output_dim (int): Length of embedding vectors.
    embeddings_initializer: Initializer for the `embeddings`
      matrix (see `keras.initializers`).
    embeddings_regularizer: Regularizer function applied to
      the `embeddings` matrix (see `keras.regularizers`).
    embeddings_constraint: Constraint function applied to
      the `embeddings` matrix (see `keras.constraints`).
    combiner (str): Reduction method, ['sum', 'mean'] or None. Default None.

  When combiner is not None, supported input and their respectively output shape are:
    N-D `Tensor`: `(d1,...,dn)`, output shape: `(d1,...,dn-1,output_dim)`, N >= 2
    2-D `RaggedTensor`: `(batch_size, ragged_dim)`, output shape: `(batch_size, output_dim)`
    2-D `SparseTensor`: `(batch_size, max_hotness)`, output shape: `(batch_size, output_dim)`
  Embedding picked from last input dimension will be reduced with given combiner.
  """

  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               combiner=None,
               **kwargs):
    if 'input_shape' not in kwargs:
      kwargs['input_shape'] = (None,)
    if input_dim <= 0 or output_dim <= 0:
      raise ValueError(
          f'Both input_dim and output_dim should be positive, found {input_dim} and {output_dim}')
    if (not base_layer_utils.v2_dtype_behavior_enabled() and 'dtype' not in kwargs):
      # In TF1, the dtype defaults to the input dtype which is typically int32,
      # so explicitly set it to floatx
      kwargs['dtype'] = backend.floatx()
    # No autocast.
    kwargs['autocast'] = False
    super().__init__(**kwargs)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)
    self.embeddings_initializer_cpu = CPUInitializer(self.embeddings_initializer)
    self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.embeddings_constraint = constraints.get(embeddings_constraint)
    self.combiner = combiner

  @tf_utils.shape_type_conversion
  def build(self, input_shape):  # pylint: disable=unused-argument
    self.embeddings = self.add_weight(shape=(self.input_dim, self.output_dim),
                                      initializer=self.embeddings_initializer_cpu,
                                      name='embeddings',
                                      regularizer=self.embeddings_regularizer,
                                      constraint=self.embeddings_constraint,
                                      experimental_autocast=False)
    self.built = True

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.combiner is None:
      return input_shape + (self.output_dim,)
    return input_shape[:-1] + (self.output_dim,)

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    dtype = backend.dtype(inputs)
    if dtype not in ['int64', 'int32']:
      inputs = tf.cast(inputs, 'int32')
    # For needed case, compute output shape and replace leading possible None with -1
    out_shape = None
    if len(inputs.shape) != 2:
      out_shape = [-1] + list(self.compute_output_shape(inputs.shape))[1:]
    # check for unsupported cases and reshape non-2D dense inputs
    if isinstance(inputs, ragged_tensor.RaggedTensor):
      if len(inputs.shape) > 2:
        raise ValueError('Ragged input should be 2D. Nested ragged is not supported.')
    else:
      if len(inputs.shape) == 1:
        if self.combiner is not None:
          raise ValueError('1D input with combiner is ambiguous. Please create batch dimension.')
        inputs = tf.reshape(inputs, [-1, 1])
      if len(inputs.shape) > 2:
        inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
    out = embedding_lookup_ops.embedding_lookup(self.embeddings, inputs, combiner=self.combiner)
    if out_shape is not None:
      out = tf.reshape(out, out_shape)
    return out

  def get_config(self):  # pylint: disable=missing-function-docstring
    config = {
        'input_dim': self.input_dim,
        'output_dim': self.output_dim,
        'embeddings_initializer': initializers.serialize(self.embeddings_initializer),
        'embeddings_regularizer': regularizers.serialize(self.embeddings_regularizer),
        'activity_regularizer': regularizers.serialize(self.activity_regularizer),
        'embeddings_constraint': constraints.serialize(self.embeddings_constraint),
        'combiner': self.combiner
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer from its config.
    Overriding this to enable instatiating fast embedding from keras embedding configs
    """
    config.pop('mask_zero', None)
    config.pop('input_length', None)
    return super().from_config(config)


class ConcatOneHotEmbedding(tf.keras.layers.Layer):
  """Concatenated one hot embedding

  Args:
    feature_sizes (list): A list of integer indicating number of features of each embedding table
    embedding_width (int): Width of embedding vector

  """

  def __init__(self, feature_sizes, embedding_width):
    super().__init__(dtype=tf.float32)
    self.embedding_width = embedding_width
    self._offsets_np = np.array([0] + feature_sizes).cumsum()

    self.params = self.add_weight("params",
                                  shape=[self._offsets_np[-1], self.embedding_width],
                                  dtype=tf.float32)
    self.offsets = tf.constant(self._offsets_np, dtype=tf.int32)

  def call(self, inputs):
    assert inputs.shape[1] == len(self.offsets) - 1

    offset_indices = inputs + self.offsets[:-1]
    embedding_out = tf.gather(params=self.params, indices=offset_indices, axis=None)

    return embedding_out
