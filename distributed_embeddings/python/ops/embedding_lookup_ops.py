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
"""Embedding ops."""

import tensorflow as tf

from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import resource_loader
from tensorflow.python.ops import resource_variable_ops

ops = tf.load_op_library(resource_loader.get_path_to_datafile('_embedding_lookup_ops.so'))


def read_var_no_copy(res_var):
  resource_variable_ops.variable_accessed(res_var)
  return ops.read_variable_no_copy(res_var.handle, dtype=res_var.dtype)


@tf.RegisterGradient("ReadVariableNoCopy")
def _read_grad(_, grad):
  """Gradient for read op(no copy)."""
  return grad


def embedding_lookup(param, ids, combiner=None):
  """Looks up embeddings for the given `ids` from a embedding tensor.

  Args:
    param (Tensor): A single tensor representing the complete embedding tensor.
    ids (Tensor or RaggedTensor): A 2D `int32` or `int64` `Tensor` containing
      the ids to be looked up in `param`.
    combiner (string or None): Reduction method, ['sum', 'mean'] or None. Default None.

  Returns:
    Tensor: A `Tensor` with the same type as the tensors in `param`.

  .. note::
    When combiner is None, returned tensor has shape: ``shape(ids) + shape(param)[1]``

    Otherwise, embedding from same row is reduced and returned tensor has shape:
    ``shape(ids)[0] + shape(param)[1]``

  Note when ids is RaggedTensor, its values and row_splits are col_index and row_index
  of CSR format hotness matrix, thus can be directly constructed.

  Raises:
    TypeError: If `param` is empty.
    ValueError: If `ids` is not 2D tensor.
  """
  if not tf.is_tensor(param):
    raise TypeError("param must be Tensor")
  if ids.get_shape().ndims != 2:
    raise ValueError("Only support 2D input")
  if combiner is None:
    return tf.nn.embedding_lookup(param, ids)
  if isinstance(ids, ragged_tensor.RaggedTensor):
    return ops.embedding_lookup_variable_hotness(read_var_no_copy(param), ids.values,
                                                 ids.row_splits, combiner)
  return ops.embedding_lookup_constant_hotness(read_var_no_copy(param), ids, combiner)


@tf.RegisterGradient("EmbeddingLookupConstantHotness")
def _embedding_lookup_constant_hotness_grad(op, grad):
  """The gradients for `embedding_lookup_constant_hotness`.
  Args:
    op (object): The `embedding_lookup_constant_hotness` `Operation` that we are differentiating,
      which we can use to find the inputs and outputs of the original op.
    grad (Tensor): Gradient with respect to the output of `embedding_lookup_constant_hotness`.
  Returns:
    IndexedSlices: A `IndexedSlices` contain sparse gradients with respect to
      the embedding parameter of `embedding_lookup_constant_hotness`.
  """
  param_shape = tf.shape(op.inputs[0])
  ids = op.inputs[1]
  grad_param_value = ops.embedding_lookup_constant_hotness_grad(grad,
                                                                ids,
                                                                combiner=op.get_attr('combiner'))

  return (tf.IndexedSlices(grad_param_value, tf.reshape(ids, [-1]), param_shape), None)


@tf.RegisterGradient("EmbeddingLookupVariableHotness")
def _embedding_lookup_variable_hotness_grad(op, grad):
  """The gradients for `embedding_lookup_variable_hotness`.
  Args:
    op (object): The `embedding_lookup_variable_hotness` `Operation` that we are differentiating,
      which we can use to find the inputs and outputs of the original op.
    grad (Tensor): Gradient with respect to the output of `embedding_lookup_variable_hotness`.
  Returns:
    IndexedSlices: A `IndexedSlices` contain sparse gradients with respect to
      the embedding parameter of `embedding_lookup_variable_hotness`.
  """
  ids = op.inputs[1]
  offsets = op.inputs[2]
  grad_param_value = ops.embedding_lookup_variable_hotness_grad(grad,
                                                                ids,
                                                                offsets,
                                                                combiner=op.get_attr('combiner'))

  param_shape = tf.cast(op.inputs[0].shape, dtype=tf.int64)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
  return (tf.IndexedSlices(grad_param_value, ids, param_shape), None, None)
