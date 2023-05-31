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
    ids (Tensor): A 2D `int32` or `int64` `Tensor` containing the ids to be looked up
      in `param`. Also support `RaggedTensor` and `SparseTensor`.
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
    # assuming no empty sample. tf.shape may fail on earlier tf version with ragged input
    try:
      dim_0 = tf.shape(ids, out_type=tf.int32)[0] if ids.shape[0] is None else ids.shape[0]
    except:  # pylint: disable=bare-except
      dim_0 = tf.shape(ids.row_splits,
                       out_type=tf.int32)[0] - 1 if ids.shape[0] is None else ids.shape[0]
    num_input = tf.shape(
        ids.values, out_type=tf.int32)[0] if ids.values.shape[0] is None else ids.values.shape[0]
    if dim_0 == num_input:
      return tf.nn.embedding_lookup(param, ids.values)
    return ops.embedding_lookup_variable_hotness(read_var_no_copy(param), ids.values,
                                                 ids.row_splits, combiner)
  if isinstance(ids, tf.SparseTensor):
    # sparse is ordered but may not be right-ragged. so we generate offset here
    # avoid d2h copy in eager mode by using sparsetensor's shape directly
    dim_0 = tf.shape(ids, out_type=tf.int32)[0] if ids.shape[0] is None else ids.shape[0]
    num_input = tf.shape(
        ids.values, out_type=tf.int32)[0] if ids.values.shape[0] is None else ids.values.shape[0]
    if dim_0 == num_input:
      return tf.nn.embedding_lookup(param, ids.values)
    # use custom op to avoid bad XLA bahavior and d2h copy caused by searchsorted
    row_splits = ops.row_to_split(ids.indices, dim_0)
    # we really want ids.values and row_splits to be same dtype to simplify things
    # since max(row_splits) here is likely ~total hotness, int32 should be ok
    # TODO(Deyu): fuse this cast into above row_to_split function and make always int32
    return ops.embedding_lookup_variable_hotness(read_var_no_copy(param), ids.values,
                                                 tf.cast(row_splits, dtype=ids.values.dtype),
                                                 combiner)
  dim1 = tf.shape(ids, out_type=tf.int32)[1] if ids.shape[1] is None else ids.shape[1]
  if dim1 == 1:
    return tf.nn.embedding_lookup(param, tf.squeeze(ids, [1]))
  if combiner == 'sum':
    return tf.reduce_sum(tf.nn.embedding_lookup(param, ids), axis=1)
  return tf.reduce_mean(tf.nn.embedding_lookup(param, ids), axis=1)


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
  param_shape = tf.shape(op.inputs[0])
  flat_ids = tf.reshape(op.inputs[1], [-1])
  offsets = op.inputs[2]
  unique_ids, unique_grad = ops.embedding_lookup_variable_hotness_grad(
      flat_ids, offsets, grad, op.inputs[0], combiner=op.get_attr('combiner'))

  return (tf.IndexedSlices(unique_grad, unique_ids, param_shape), None, None)


def integer_lookup(table, count, keys, capacity):
  resource_variable_ops.variable_accessed(table)
  resource_variable_ops.variable_accessed(count)
  return ops.integer_lookup(table.handle, count.handle, keys, capacity, count.dtype)
