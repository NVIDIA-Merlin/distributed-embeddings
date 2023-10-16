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
"""Distributed Embedding layers and utils"""
import types
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.python.keras.utils import tf_utils
import horovod
import horovod.tensorflow as hvd
import horovod.tensorflow.keras as hvd_keras
from distributed_embeddings.python.ops.embedding_lookup_ops import read_var_no_copy
from .embedding import Embedding


class ConcatInitializer(tf.keras.initializers.Initializer):
  """ initializer wrapper to handle automatic concat table on first dimension
  """

  def __init__(self, initializer, sizes):
    self._initializer = initializer
    self.sizes = sizes

  def __call__(self, shape, dtype=None, **kwargs):
    weights = [self._initializer([size, shape[1]], dtype=dtype, **kwargs) for size in self.sizes]
    weights = tf.concat(weights, axis=0)
    return weights


def _get_shape(tensor):
  """Return shape of tensor

  Static shape is not always available, in which case we use tf.shape to get dynamic shape

  Args:
      tensor (Tensor): Input tensor

  Returns:
      tf.Shape
  """
  if tensor.shape is not None and None not in tensor.shape:
    return tensor.shape
  return tf.shape(tensor)


def _argsort(l, key=None, reverse=False):
  if key is None:
    key = lambda x: x

  r = list(sorted(enumerate(l), key=lambda x: key(x[1]), reverse=reverse))
  order = [x[0] for x in r]
  values = [x[1] for x in r]
  return values, order


def _transpose_ragged_2d(x, world_size, num_local_features):
  """
  Transpose from a tf.RaggedTensor from worker-major to feature-major format.

  Args:
    x (tf.RaggedTensor): input tensor
    world_size: total number of workers
    num_local_features: number of features on the current worker

  Returns:
    A transposed tf.RaggedTensor
  """
  x = tf.split(x, world_size * num_local_features)
  transposed = []
  for j in range(num_local_features):
    for i in range(world_size):
      transposed.append(x[i * num_local_features + j])
  transposed = tf.concat(transposed, axis=0)
  return transposed


def _dp_to_mp_input_ragged(dp_inputs, rank_to_local_features):
  """
  Transforms embedding input indices from data-parallel to model-parallel paradigm
  using horovod all-to-all operations. Only supports tf.RaggedTensor inputs.

  Args:
    dp_inputs (dict of ragged tf.Tensors): a dictionary mapping feature index
      to a ragged data-parallel tensor with the input data for this feature.
    rank_to_local_features (dict of lists): a dictionary mapping the rank of each horovod worker
        to the list of feature indices that are supposed to be gathered onto that worker.

  Returns:
    A dictionary mapping the index of the feature to a ragged tensor.
    Each tensor contains model-parallel data for the corresponding feature.

  Raises:
    ValueError: in case of incorrect input.
  """

  if not isinstance(dp_inputs, dict):
    raise ValueError(f'Expected a dict, got: {type(dp_inputs)}')

  if not dp_inputs:
    return {}

  # The split tensor for the first all-to-all of the flat tensor values
  a2a_splits = {}

  # features_dp is a list of all values we need to send to each worker.
  # It is not simply dp_inputs.values() because some tensors might need to be sent
  # to multiple workers. For example because an embedding table is split onto multiple workers.
  features_dp = []
  for worker, features in rank_to_local_features.items():
    a2a_splits[worker] = tf.math.reduce_sum([tf.size(dp_inputs[f].flat_values) for f in features])
    # Need to cast in case of zero, which turns out as float32.
    a2a_splits[worker] = tf.cast(a2a_splits[worker], dtype=tf.int32)

    for feature_id in features:
      features_dp.append(dp_inputs[feature_id])

  flat_values_a2a_splits = [a2a_splits[i] for i in range(hvd.size())]

  flat_features_dp = tf.concat(features_dp, axis=0).flat_values
  # Perform the first all-to-all for the ragged tensor values.
  flat_features_mp, _ = hvd.alltoall(flat_features_dp, splits=flat_values_a2a_splits)

  local_batch = next(iter(dp_inputs.values())).shape[0]

  # Compute the splits for the second all-to-all (row-lengths).
  row_lengths_a2a_splits = [local_batch * len(rank_to_local_features[r]) for r in range(hvd.size())]

  # Perform the second all-to-all on the row-length data.
  row_lengths_dp = tf.concat([x.row_lengths() for x in features_dp], axis=0)
  row_lengths_mp, _ = hvd.alltoall(row_lengths_dp, splits=row_lengths_a2a_splits)

  num_mp_features = len(rank_to_local_features[hvd.rank()])
  if num_mp_features == 0:
    # The current worker has not received any variable-length features.
    # Nothing to be done. Simply return an empty dict.
    return {}

  # This reshape is necessary for correct static shape inference in graph mode
  row_lengths_mp = tf.reshape(row_lengths_mp, shape=[num_mp_features * local_batch * hvd.size()])

  # Construct a large ragged tensor containing all the features the current worker has received.
  flat_features_mp = tf.RaggedTensor.from_row_lengths(values=flat_features_mp,
                                                      row_lengths=row_lengths_mp)

  # Need to transpose from worker-major to feature-major before we split into individual features.
  flat_features_mp = _transpose_ragged_2d(flat_features_mp,
                                          num_local_features=num_mp_features,
                                          world_size=hvd.size())

  # Split the large tensor into a list of smaller ragged tensors for each feature.
  features_mp = tf.split(flat_features_mp, num_or_size_splits=num_mp_features)
  features_mp = dict(zip(rank_to_local_features[hvd.rank()], features_mp, strict=True))
  return features_mp


def _dp_to_mp_input_dense(dp_inputs, rank_to_local_features):
  """
    Transforms embedding input indices from data-parallel to model-parallel paradigm
    using horovod all-to-all operations. Only supports fixed-length tf.Tensor inputs.

  Args:
    dp_inputs (dict of dense tf.Tensors): a dictionary mapping feature index
        to a potentially ragged data-parallel tensor with the input data for this feature.
    rank_to_local_features (dict of lists): a dictionary mapping the rank of each horovod worker
        to the list of feature indices that are supposed to be gathered onto that worker.

  Returns:
    A dictionary mapping the index of the feature to a dense tf.Tensor.
    Each tensor contains model-parallel data for the corresponding feature.

  Raises:
    ValueError: in case of incorrect input.
  """
  if not isinstance(dp_inputs, dict):
    raise ValueError(f'Expected a dict, got: {type(dp_inputs)}')

  if not dp_inputs:
    return {}

  world_size, rank = hvd.size(), hvd.rank()

  comm_dtype = tf.int32
  for inp in dp_inputs.values():
    if inp.dtype == tf.int64:
      comm_dtype = tf.int64
  dp_inputs = {k: tf.cast(v, comm_dtype) for k, v in dp_inputs.items()}
  local_shapes, local_splits, global_splits, flat_inputs = [], [], [], []

  for rank_input_ids in rank_to_local_features.values():
    rank_inputs = [dp_inputs[feature_key] for feature_key in rank_input_ids]
    local_shapes.append([_get_shape(inp) for inp in rank_inputs])
    rank_inputs = [tf.reshape(inp, [-1]) for inp in rank_inputs]
    local_splits.append([_get_shape(inp)[0] for inp in rank_inputs])
    global_splits.append(sum(local_splits[-1]))
    flat_inputs += rank_inputs
  dp_inputs = tf.concat(flat_inputs, 0)

  mp_inputs, _ = hvd.alltoall(dp_inputs, splits=global_splits, name='inp_dp_to_mp')

  mp_inputs = tf.reshape(mp_inputs, [world_size, -1])
  mp_inputs = tf.split(mp_inputs, local_splits[rank], 1)
  mp_inputs = [
      tf.reshape(inp, [world_size * shape[0]] + shape[1:])
      for inp, shape in zip(mp_inputs, local_shapes[rank])
  ]

  mp_inputs = dict(zip(rank_to_local_features[rank], mp_inputs, strict=True))
  return mp_inputs


def _dp_to_mp_input(dp_inputs, rank_to_local_features):
  """
  Transforms embedding input indices from data-parallel to model-parallel paradigm
  using horovod all-to-all operations. The resulting model-parallel indices
  can be used to run a model-parallel embedding lookup. Supports dense tf.Tensor
  and tf.RaggedTensor inputs.

  Args:
    dp_inputs (dict of potentially ragged tf.Tensors): a dictionary mapping feature index
        to a potentially ragged data-parallel tensor with the input data for this feature.
        Each tensor can be either a dense tf.Tensor or a tf.RaggedTensor.
    rank_to_local_features (dict of lists): a dictionary mapping the rank of each horovod worker
        to the list of feature indices that are supposed to be gathered onto that worker.
        E.g., passing rank_to_local_features={0: [0, 1], 1: [2]} means that worker "0" should
        receive the data for features "0" and "1" and worker "1" should receive
        the data for feature "2".

  Returns:
    A dictionary mapping the index of the feature to a potentially ragged tensor.
    Each tensor contains model-parallel data for the corresponding feature.

  Raises:
    ValueError: if a tf.SparseTensor input has been passed. This is not supported.
  """
  # Handle the trivial single-worker case separately.
  if hvd.size() <= 1:
    # Expected input order may still change in case of single process.
    inputs = {idx: dp_inputs[idx] for idx in rank_to_local_features[0]}
    return inputs

  if isinstance(dp_inputs, list):
    dp_inputs = dict(enumerate(dp_inputs))

  # Partition the input tensors into two groups: fixed length and variable length,
  # so that they can be handled separately.
  ragged_dp_inputs, dense_dp_inputs = {}, {}
  for i, f in dp_inputs.items():
    if isinstance(f, tf.RaggedTensor):
      ragged_dp_inputs[i] = f
    elif isinstance(f, tf.SparseTensor):
      # TODO(tgrel): support sparse input, possibly by converting to ragged here
      raise ValueError('Sparse tensor data-parallel input is not supported')
    else:
      dense_dp_inputs[i] = f

  # Partition the input map into two separate maps:
  #  - One that maps each worker rank to a list of dense features it is supposed to receive.
  #  - The other that maps each worker rank to a list of variable length features it
  #    is supposed to receive.
  rank_to_dense_features, rank_to_ragged_features = {}, {}
  for rank in range(hvd.size()):
    all_rank_features = rank_to_local_features[rank]
    rank_to_dense_features[rank] = [v for v in all_rank_features if v in dense_dp_inputs]
    rank_to_ragged_features[rank] = [v for v in all_rank_features if v in ragged_dp_inputs]

  # Call the subroutine for fixed-length inputs.
  dense_mp_inputs = _dp_to_mp_input_dense(dense_dp_inputs,
                                          rank_to_local_features=rank_to_dense_features)

  # Call the subroutine for variable-length inputs.
  ragged_mp_inputs = _dp_to_mp_input_ragged(ragged_dp_inputs,
                                            rank_to_local_features=rank_to_ragged_features)

  mp_inputs = {**dense_mp_inputs, **ragged_mp_inputs}
  return mp_inputs


@tf.custom_gradient
def grouped_reducescatter_unscaled(inputs):
  outputs = hvd.grouped_reducescatter(inputs, op=hvd.Sum)

  def grad(*upstream):
    return hvd.grouped_allgather(upstream)

  return outputs, grad


class DistEmbeddingStrategy():
  """Distributed embedding strategy

  Args:
    embeddings (list of Embedding): list of unbuilt Embedding layers globally
    strategy (str): A string indicates how embedding tables are distributed.
        Choices are [“basic”, “memory_balanced”]. Default "basic"
    input_table_map (list or None): A list of table ids mapping each input to a table, i.e.,
        `input[i]` map to `table[input_table_map[i]]`. None means there are same number of
        inputs/tables and `input[i]` map to `table[i]`. Default None.
    column_slice_threshold: desired upper bound of elements count in each slice.
    row_slice_threshold: desired lower bound where table larger than this will be row sliced.
    gpu_embedding_size: total number of local table-parallel embedding elements to fit on the GPU.
                        if more elements are used, they will be CPU offloaded.
                        Set to None (default) to try to fit all table-parallel tables on the GPU.
                        CPU offloading row-sliced and data-parallel tables is not supported.
    data_parallel_threshold: Embedding smaller than this will be run data-parallel.

  Attributes:
    strategy: string indicates how embedding tables are distributed.
    column_slice_threshold: desired upper bound of elements count in each slice.
    row_slice_threshold: desired lower bound where table larger than this will be row sliced.
    gpu_embedding_size: total number of local table-parallel embedding elements to fit on the GPU.
                        if more elements are used, they will be CPU offloaded.
                        Set to None (default) to try to fit all table-parallel tables on the GPU.
                        CPU offloading row-sliced and data-parallel tables is not supported.
    data_parallel_threshold: Embedding smaller than this will be run data-parallel.
    sliced_out_ranges: list of [output_pos, num_slices] for each input, used to merged outputs.
    table_groups: lists of table ids. group 0 runs dp, group 1 do column slice with table parallel,
                  group 2 runs row_slice onto all workers.
    input_groups: input ids coresponding to table grouping.
    map_groups: input to table map within each group.
    rev_group_ids: list used to reorder grouped output back into flat input order.
    row_sliced_configs: configs that's been row sliced.
    row_inputs_offsets: add to row slice input so inputs except certain range will go OOB.
    input_ids_list: nested list, contain input index list in rank order. use for dp_input == False.
    local_maps: nested list, contain per rank local input to table map.
    local_configs: nested list, contain per rank lists of local table configs.
    local_input_offsets: nested list, contain per rank offsets to form input for concat embedding.
    local_weight_offsets: nested list, contain per rank weight internal offsets get/set_weights.
    local_group_list: nested list, contain per rank concat grouping info for get/set_weights.
    table_ids: nested list, contain per rank table ids for get/set_weights.
    widths_list_flat: list of all output width, before merging slices and in worker order
    rev_tp_ids: list used to reorder table parallel output back into flat tp input order.
  """

  def __init__(self,
               embeddings,
               world_size,
               strategy="basic",
               input_table_map=None,
               column_slice_threshold=None,
               row_slice_threshold=None,
               data_parallel_threshold=None,
               gpu_embedding_size=None):
    # code in DMP to skip hvd call in single process case may assume "basic"
    self.strategy = "basic" if world_size == 1 else strategy
    # column_slice can be used to enable more table concat, so keep it in single process
    self.column_slice_threshold = column_slice_threshold
    self.row_slice_threshold = row_slice_threshold
    self.gpu_embedding_size = gpu_embedding_size
    self.data_parallel_threshold = data_parallel_threshold
    self.global_configs = [e.get_config() for e in embeddings]
    # Insert layer type information to config dicts
    for config, embedding in zip(self.global_configs, embeddings):
      config['layer_type'] = type(embedding)
    if input_table_map is None:
      input_table_map = list(range(len(embeddings)))

    # separated table ids into groups for different strat
    self.table_groups = self.init_table_groups(self.global_configs)
    # input ids and map. rev_group_ids here to reverse grouped call back to input order
    self.input_groups, self.map_groups, self.rev_group_ids = self.init_input_and_map_groups(
        self.table_groups, input_table_map)

    # 1. handle data parallel
    self.dp_configs = [self.global_configs[idx] for idx in self.table_groups[0]]

    # 2. handle row slicing
    if self.table_groups[2]:
      self.row_sliced_configs, self.row_inputs_offsets = self.create_row_sliced_configs(
          [self.global_configs[idx] for idx in self.table_groups[2]], world_size)
    else:
      self.row_sliced_configs = [[] for _ in range(world_size)]

    # 3. handle column slicing and table parallel
    if not self.table_groups[1]:
      return

    # Create (maybe) sliced configs
    sliced_configs, self.sliced_out_ranges = self.create_col_sliced_configs(
        [self.global_configs[idx] for idx in self.table_groups[1]], world_size,
        self.column_slice_threshold, self.map_groups[1])

    # Apply strategy and save nested list containing table indices by rank
    table_ids = self.apply_strategy(self.strategy, world_size, sliced_configs)

    # Following are ALL nested lists holding info for distributing embeddings, ordered by rank
    self.input_ids_list = []
    self.local_maps = []
    self.local_configs = []
    self.local_input_offsets = []
    self.local_weight_offsets = []
    self.local_group_list = []
    self.table_ids = []

    # Each worker loop over all rank to get global view of strategy
    for rank_table_ids in table_ids:
      # first merge different shards of same table that ends up on same rank
      rank_table_ids, rank_configs = self._merge_slices(rank_table_ids, sliced_configs)
      self.table_ids.append(rank_table_ids)

      # calculate local input ids and map from this rank's table_ids and global input map
      rank_input_ids, rank_input_map = [], []
      for m, table_idx in enumerate(rank_table_ids):
        for k, mapped_idx in enumerate(self.map_groups[1]):
          if table_idx == mapped_idx:
            rank_input_ids.append(k)
            rank_input_map.append(m)

      # Only offloading the table-parallel/column-sliced embeddings.
      rank_configs = self._maybe_offload(configs=rank_configs)

      # concat eligible tables then adjust local config and map
      rank_configs, rank_input_map, input_offsets, group, weight_offsets = self._create_concat(
          rank_configs, rank_input_map)

      # save results to global nested list
      self.input_ids_list.append(rank_input_ids)
      self.local_configs.append(rank_configs)
      self.local_maps.append(rank_input_map)
      self.local_input_offsets.append(input_offsets)
      self.local_group_list.append(group)
      self.local_weight_offsets.append(weight_offsets)

    # create a flatten list contain table widths, in worker order, used for slice after alltoall
    # This is fast but might switch to not use this to support non-2D output(no local combiner)
    self.widths_list_flat = []
    for config, input_map in zip(self.local_configs, self.local_maps):
      self.widths_list_flat += [config[m]['output_dim'] for m in input_map]

    # List of indices to shuffle worker ordered embedding outputs back to original order
    worker_order_input_ids = [item for sublist in self.input_ids_list for item in sublist]
    self.rev_tp_ids = [
        index
        for _, index in sorted(zip(worker_order_input_ids, range(len(worker_order_input_ids))))
    ]

  def _maybe_offload(self, configs):
    """
      Offloads the largest tables among the "configs" argument,
      such that the sum of not offloaded tables is smaller than the threshold.

      Args:
        configs (List): a list of configs to process

      Returns:
        A list of configs in the same order as the "configs" argument,
        but with the "cpu_offload" field set to either True or False.
    """
    configs = configs.copy()

    if self.gpu_embedding_size is None:
      for config in configs:
        config['cpu_offload'] = False
      return configs

    current_total_size = 0

    # use indices rather than sorting the list directly to maintain the original order
    _, order = _argsort(configs, key=lambda x: x['input_dim'] * x['output_dim'])
    for index in order:
      config = configs[index]
      current_total_size += config['input_dim'] * config['output_dim']
      config['cpu_offload'] = current_total_size > self.gpu_embedding_size
    return configs

  # below are the methods to divide table into groups and adjust input and input map accordingly
  def init_table_groups(self, configs):
    # We want to support data parallel, table parallel, column slice and row slice
    # Couple assumptions:
    # - strat applied by size in above order, meaning small table run dp -> large table run row slice
    # - currently only apply one of above to any table. it may make sense to mix row/col slice in future
    # - because communication pattern is different, we run 3 separate groups calls
    # - non-symmetric table parallel is only applied to column sliced group
    num_elems = [config['input_dim'] * config['output_dim'] for config in configs]
    dp, col, row = [], [], []
    for i, num_elem in enumerate(num_elems):
      if self.data_parallel_threshold and num_elem <= self.data_parallel_threshold:
        dp.append(i)
      elif self.row_slice_threshold and num_elem >= self.row_slice_threshold:
        row.append(i)
      else:
        col.append(i)
    return [dp, col, row]

  def init_input_and_map_groups(self, table_groups, input_table_map):
    dp, col, row = table_groups
    # pick out inputs for each group
    dp_in, col_in, row_in = [], [], []
    dp_map, col_map, row_map = [], [], []
    for i, idx in enumerate(input_table_map):
      if idx in dp:
        dp_in.append(i)
        dp_map.append(dp.index(idx))
      elif idx in col:
        col_in.append(i)
        col_map.append(col.index(idx))
      elif idx in row:
        row_in.append(i)
        row_map.append(row.index(idx))
      else:
        raise ValueError("Wrong input initializing input/map groups.")
    flat_input_ids = dp_in + col_in + row_in
    reverse_ids = [index for _, index in sorted(zip(flat_input_ids, range(len(flat_input_ids))))]
    return [dp_in, col_in, row_in], [dp_map, col_map, row_map], reverse_ids

  def maybe_slice_table_column(self, orig_config, column_slice_threshold, world_size):
    """Column slice a embedding config if size exceed column_slice_threshold.
    Assume N is smallest power of 2 so that when evenly slice original table into N tables,
    each have less than column_slice_threshold elements.
    So final number of slices will be min(N, world_size, table_width).
    Args:
      orig_config (dict): embedding layer config to create slices from
      column_slice_threshold (int or None): desired upper bound of elements count in each slice
      world_size (int): number of total model parallel worker
    Returns:
      sliced_config (list): list of embedding layer config that concat into original config
    """
    if column_slice_threshold is None:
      column_slice_threshold = float('inf')
    table_size = orig_config['input_dim'] * orig_config['output_dim']
    num_slices = 1
    while table_size > column_slice_threshold:
      num_slices *= 2
      table_size /= 2
    if num_slices == 1:
      return [orig_config.copy()]
    num_slices = min(num_slices, world_size, orig_config['output_dim'])
    column_per_slice = orig_config['output_dim'] // num_slices
    remainder = orig_config['output_dim'] % num_slices
    sliced_config = []
    for i in range(num_slices):
      config = orig_config.copy()
      config['output_dim'] = column_per_slice
      if i < remainder:
        config['output_dim'] += 1
      sliced_config.append(config)
    return sliced_config

  def create_col_sliced_configs(self, global_col_configs, world_size, column_slice_threshold,
                                input_table_map):
    """Create column sliced configs from global configs.
    This function also calculate ranges of data parallel output needs concat due to this slice.
    Args:
      global_col_configs (list): selected configs for doing column slice
      world_size (int): number of model parallel workers
      column_slice_threshold (int or None): desired upper bound of elements count in each slice
      input_table_map (list): A list of table ids mapping each input to a table
    Returns:
      sliced_configs (list): same length as global configs. each element is a list represent sliced
    form of global config at the same position.
      sliced_out_ranges (list): each element is list of 2 integers, representing output ranges need
    to be concatenated to re-form output due to above slice.
    """
    # less table than worker, we try our best to slice into worker count slices(may go over)
    if column_slice_threshold is None:
      table_sizes = [config['input_dim'] * config['output_dim'] for config in global_col_configs]
      while world_size > len(table_sizes):
        table_sizes.sort()
        column_slice_threshold = table_sizes[-1] - 1
        cur_max_size = table_sizes.pop(-1)
        table_sizes += [cur_max_size // 2, cur_max_size // 2]

    sliced_configs = []
    for col_config in global_col_configs:
      maybe_sliced_config = self.maybe_slice_table_column(col_config, column_slice_threshold,
                                                          world_size)
      sliced_configs.append(maybe_sliced_config)
    # figure out ranges of output that needs concat
    # this needs to be in output order, otherwise range modification would fail
    sliced_out_ranges = []
    for input_id, table_id in enumerate(input_table_map):
      if len(sliced_configs[table_id]) > 1:
        sliced_out_ranges.append([input_id, input_id + len(sliced_configs[table_id])])
    return sliced_configs, sliced_out_ranges

  def create_row_sliced_configs(self, global_row_configs, world_size):
    # initial test code. not considering corner cases
    sliced_configs, offsets = [], []
    for orig_config in global_row_configs:
      sliced_config, offset = [], []
      cur_offset = 0
      row_per_slice = orig_config['input_dim'] // world_size
      remainder = orig_config['input_dim'] % world_size
      for i in range(world_size):
        config = orig_config.copy()
        config['input_dim'] = row_per_slice
        if i < remainder:
          config['input_dim'] += 1
        sliced_config.append(config)
        offset.append(cur_offset)
        cur_offset -= config['input_dim']
      sliced_configs.append(sliced_config)
      offsets.append(offset)
    # re-divide lists by rank
    sliced_configs = [list(rank_configs) for rank_configs in zip(*sliced_configs)]
    offsets = [list(rank_offsets) for rank_offsets in zip(*offsets)]
    return sliced_configs, offsets

  # pylint: disable=missing-param-doc,missing-type-doc,missing-raises-doc
  def apply_strategy(self, mode, world_size, sliced_configs):
    """Distribute tables to workers from sliced config, a nested list.
    Returns:
      divided_ids (list): world_size length list. Each element is list of
    sliced table ids distribute to rank according to position.
    """
    global_ids = []
    table_sizes = []
    for i, sliced_config in enumerate(sliced_configs):
      for config in sliced_config:
        global_ids.append(i)
        table_sizes.append(config['input_dim'] * config['output_dim'])

    # Round-robin distribute tables onto workers
    if mode == 'basic':
      divided_ids = [global_ids[i::world_size] for i in range(world_size)]
    # Distributed table so that memory is balanced while table count remain even
    elif mode == 'memory_balanced':
      sorted_ids = [idx for _, idx in sorted(zip(table_sizes, global_ids), reverse=True)]
      divided_ids = [
          sorted_ids[i::2 * world_size] + sorted_ids[(2 * world_size - 1 - i)::2 * world_size]
          for i in range(world_size)
      ]
    # Try to optimize for total memory first. After sorted by size, table are distributed one by one
    # to worker with lowest total size. Memory usage will be more even but table count may not.
    elif mode == 'memory_optimized':
      sorted_pairs = list(sorted(zip(table_sizes, global_ids)))
      res = [[0, []] for _ in range(world_size)]
      while sorted_pairs:
        cur = sorted_pairs.pop()
        res[0][0] += cur[0]
        res[0][1].append(cur[1])
        res = sorted(res)
      divided_ids = [r[1] for r in res]
    else:
      raise ValueError(F"Unsupported strategy {strategy}")
    return divided_ids

  # Concat table so different table now become shared embedding. XLA does rest of optimization.
  def _create_concat(self, table_configs, input_maps):
    # first get local table id into groups
    grouped_table_ids, concat_configs = [], []
    for table_id, config in enumerate(table_configs):
      for group, concat_config in zip(grouped_table_ids, concat_configs):
        same_output_dim = config['output_dim'] == concat_config['output_dim']
        same_combiner = config.get('combiner') == concat_config.get('combiner')
        no_offload = not (config['cpu_offload'] or concat_config['cpu_offload'])
        if same_output_dim and same_combiner and no_offload:
          group.append(table_id)
          concat_config['input_dim'] += config['input_dim']
          concat_config['input_dims'].append(config['input_dim'])
          concat_config['offsets'].append(concat_config['offsets'][-1] + config['input_dim'])
          break
      else:  # can't merge with any group, create a new one
        grouped_table_ids.append([table_id])
        config['input_dims'] = [config['input_dim']]
        config['offsets'] = [0, config['input_dim']]
        concat_configs.append(config)

    # adjust input map and create according offset map
    new_input_map, input_offsets = [], []
    for input_map in input_maps:
      for gid, (group, concat_config) in enumerate(zip(grouped_table_ids, concat_configs)):
        if input_map in group:
          new_input_map.append(gid)
          input_offsets.append(concat_config['offsets'][group.index(input_map)])
          break

    # switch to concat initializer to keep behavior associated with shape
    for concat_config in concat_configs:
      input_dims = concat_config.pop('input_dims')
      if len(input_dims) > 1:
        # TODO(deyuf): we don't really need serialize and can just get from original class
        if 'embeddings_initializer' in concat_config:
          orig_initializer = initializers.deserialize(concat_config['embeddings_initializer'])
          concat_config['embeddings_initializer'] = ConcatInitializer(orig_initializer, input_dims)

    # record weight offsets for get/set.
    weight_offsets = [concat_config.pop('offsets', None) for concat_config in concat_configs]
    return concat_configs, new_input_map, input_offsets, grouped_table_ids, weight_offsets

  # Helper function to re-merge slices of same table in cases they end up on same workers
  def _merge_slices(self, rank_table_ids, sliced_configs):
    merged_table_ids, rank_configs = [], []
    for table_idx in rank_table_ids:
      # this id has been seen on this rank before, merge it with earlier shard
      if table_idx in merged_table_ids:
        config_to_merge = sliced_configs[table_idx].pop(0)
        index_to_merge = merged_table_ids.index(table_idx)
        rank_configs[index_to_merge]['output_dim'] += config_to_merge['output_dim']
        # modify output concat ranges
        for out_range in self.sliced_out_ranges:
          if out_range[0] == table_idx:
            out_range[-1] -= 1
      else:
        merged_table_ids.append(table_idx)
        rank_configs.append(sliced_configs[table_idx].pop(0))
    return merged_table_ids, rank_configs


class DistributedEmbedding(tf.keras.layers.Layer):
  """Distributed embedding wrapper

  This class is a hybrid parallel wrapper around embedding. It handles all to all communication of
  forward and backward of embedding.

  Args:
    embeddings (list of keras Embedding layers): embedding tables to be distributed
    strategy (str): A string indicates how embedding tables are distributed.
        Choices are [“basic”, “memory_balanced”]. Default "basic"
    column_slice_threshold (int or None): If None, column slice only happen when there are more
        workers than tables. In that case, column_slice_threshold will be choose automatically
        so each worker receive at least one slice.
        If not None, embedding tables with more elements than column_slice_threshold will be divide
        into N even pieces alone embedded width dimension.
        N is smallest power of 2 makes each slice smaller than column_slice_threshold. Default None.
    row_slice_threshold: Embedding larger than this will be evenly row sliced onto all workers
    dp_input (bool): If True, takes data parallel input, i.e. in shape
        [local_batch_size x global_num_embeddings]. Otherwise take model parallel input in shape
        [global_batch_size x local_num_embeddings]. Default True.
    input_table_map (list or None): same length list as inputs, map `input[i]`
        to `table[input_table_map[i]]`. None means there are same number of
        inputs/tables and `input[i]` map to `table[i]`. Default None.
    data_parallel_threshold: Embedding smaller than this will be run data-parallel.
    gpu_embedding_size: total number of local table-parallel embedding elements to fit on the GPU.
                        if more elements are used, they will be CPU offloaded.
                        Set to None (default) to try to fit all table-parallel tables on the GPU.
                        CPU offloading row-sliced and data-parallel tables is not supported.
  """

  def __init__(self,
               embeddings,
               strategy="basic",
               column_slice_threshold=None,
               row_slice_threshold=None,
               dp_input=True,
               input_table_map=None,
               data_parallel_threshold=None,
               gpu_embedding_size=None,
               **kwargs):

    super().__init__(**kwargs)
    if strategy not in ['basic', 'memory_balanced', 'memory_optimized']:
      raise ValueError(F"Unsupported shard strategy {strategy}")

    # Currently assume data parallel ranks == model parallel ranks
    # TODO(deyuf): add more control over this with newly added hvd process_set api
    if not hvd.is_initialized():
      hvd.init()
    self.world_size = hvd.size()
    self.rank = hvd.rank()

    # single worker case fallback to no dp and no row slice.
    # ideally we could fallback to dp, but do mp for mp_input backward compatibilty
    self.dp_input = dp_input
    self.column_slice_threshold = column_slice_threshold
    self.gpu_embedding_size = gpu_embedding_size
    if self.world_size > 1:
      self.row_slice_threshold = row_slice_threshold if dp_input else None
      self.data_parallel_threshold = data_parallel_threshold if dp_input else None
    else:
      self.row_slice_threshold = None
      self.data_parallel_threshold = None

    # get model parallel distribution strategy
    self.strategy = DistEmbeddingStrategy(embeddings,
                                          self.world_size,
                                          strategy,
                                          input_table_map=input_table_map,
                                          column_slice_threshold=column_slice_threshold,
                                          row_slice_threshold=self.row_slice_threshold,
                                          data_parallel_threshold=self.data_parallel_threshold,
                                          gpu_embedding_size=self.gpu_embedding_size)

    # Here we make sure empty lists exist
    # create data parallel layers
    self.dp_layers = []
    if self.strategy.table_groups[0]:
      for config in self.strategy.dp_configs:
        self.dp_layers.append(self._create_layer_from_config(config))

    # create (maybe) column sliced embeddings and table parallel.
    self.local_embedding_layers = []
    self.col_inputs_offsets = []
    if self.strategy.table_groups[1]:
      # Handle explicit threshold or corner cases, in which worker may receive no configs
      # Column slice still need to expand all gpu, otherwise alltoall fails
      if not all(rank_configs for rank_configs in self.strategy.local_configs):
        raise ValueError("Not enough table after slicing to run on all worker."
                         "Try decrease column_slice_threshold or decrease worker count")
      for config in self.strategy.local_configs[self.rank]:
        self.local_embedding_layers.append(self._create_layer_from_config(config))
      self.col_inputs_offsets = [
          None if offset == 0 else tf.constant([offset], dtype=tf.int64)
          for offset in self.strategy.local_input_offsets[self.rank]
      ]

    # create row sliced embeddings.
    self.row_layers = []
    self.row_inputs_offsets = []
    if self.strategy.table_groups[2]:
      for config in self.strategy.row_sliced_configs[self.rank]:
        self.row_layers.append(self._create_layer_from_config(config))
      self.row_inputs_offsets = [
          None if offset == 0 else tf.constant([offset], dtype=tf.int64)
          for offset in self.strategy.row_inputs_offsets[self.rank]
      ]

  def _create_layer_from_config(self, config):
    # For stock keras Embedding, we switch underlying layer for better performance
    # If inputs are custom layers, original layer will be used
    layer_type = config.pop('layer_type')
    offloaded = config.pop('cpu_offload', False)

    if layer_type == tf.keras.layers.Embedding:
      layer_type = Embedding

    if offloaded and layer_type == Embedding:
      config['use_custom_kernel'] = False

    layer = layer_type.from_config(config)
    layer.cpu_offloaded = offloaded
    return layer

  def _call_data_parallel(self, inputs):
    outputs = [self.dp_layers[m](inp) for m, inp in zip(self.strategy.map_groups[0], inputs)]
    outputs = [tf.cast(output, self.compute_dtype) for output in outputs]

    return outputs

  def _call_table_parallel(self, inputs):  # pylint: disable=missing-param-doc,missing-type-doc
    """Call function that do embeddings and communication

    Currently, it requires same batch_size on all workers.
    """
    # get model parallel input from data parallel
    if self.dp_input:
      rank_to_local_features = dict(enumerate(self.strategy.input_ids_list))
      inputs = _dp_to_mp_input(inputs, rank_to_local_features)
      inputs = list(inputs.values())

    if len(inputs) != len(self.strategy.local_maps[self.rank]):
      raise ValueError(F"Expect {self.strategy.local_maps[self.rank]} inputs, got {len(inputs)}.")

    # offset inputs
    inputs = [
        inp if offset is None else tf.cast(inp, tf.int64) + offset
        for inp, offset in zip(inputs, self.col_inputs_offsets)
    ]
    # do embedding
    mp_outs = [
        self.local_embedding_layers[m](inp)
        for m, inp in zip(self.strategy.local_maps[self.rank], inputs)
    ]
    mp_outs = [tf.cast(output, self.compute_dtype) for output in mp_outs]

    if self.world_size > 1:
      # TODO(deyuf): current assume 2D with same batch for all output, ideally should support general case
      mp_outs = [tf.reshape(mp_out, [self.world_size, -1]) for mp_out in mp_outs]
      mp_outs = tf.reshape(tf.concat(mp_outs, axis=1), [-1])
      dp_outs = hvd.alltoall(mp_outs, name='out_mp_to_dp')
      batch_size = tf.shape(
          inputs[0], out_type=tf.int32)[0] if inputs[0].shape[0] is None else inputs[0].shape[0]
      local_bs = batch_size // self.world_size
      num_elements = [local_bs * width for width in self.strategy.widths_list_flat]
      split_outs = tf.split(dp_outs, num_elements)
      mp_outs = [tf.reshape(split_out, [local_bs, -1]) for split_out in split_outs]

    # reorder outputs to be same as inputs order
    result = [mp_outs[index] for index in self.strategy.rev_tp_ids]

    # Concat sliced outputs result from column slicing back together
    for start, end in self.strategy.sliced_out_ranges:
      result[start:end] = [tf.concat(result[start:end], axis=-1)]

    return result

  def _call_row_slice(self, inputs):
    # initial version, just allgather input, do lookup and allreduce output
    # for lookup that does not exist on this worker(OOB), zero vector is added in

    inputs = hvd.grouped_allgather(inputs)
    # offset inputs
    inputs = [
        inp if offset is None else tf.cast(inp, tf.int64) + offset
        for inp, offset in zip(inputs, self.row_inputs_offsets)
    ]
    # do embedding
    outputs = [self.row_layers[m](inp) for m, inp in zip(self.strategy.map_groups[2], inputs)]
    outputs = [tf.cast(output, self.compute_dtype, name='row_slice_cast') for output in outputs]
    outputs = grouped_reducescatter_unscaled(outputs)

    return outputs

  def set_col_slice_weights(self, weights):
    if not weights:
      return []
    if self.world_size == 1:
      if isinstance(weights[0], str):
        weights = [np.load(file=path, mmap_mode='r') for path in weights]
    else:
      slice_info = [[rank_table_id.count(table_id)
                     for rank_table_id in self.strategy.table_ids]
                    for table_id in range(len(weights))]
      local_info = [slice_info[index] for index in self.strategy.table_ids[self.rank]]
      weights = [weights[index] for index in self.strategy.table_ids[self.rank]]
      if isinstance(weights[0], str):
        weights = [np.load(file=path, mmap_mode='r') for path in weights]

      def _slice_weight_for_rank(weight, info, global_rank):
        num_columns = weight.shape[1]
        num_slices = sum(info)
        column_per_slice = num_columns // num_slices
        remainder = num_columns % num_slices
        rank = sum(info[:global_rank])

        start = column_per_slice * rank + min(rank, remainder)
        rank += 1
        end = column_per_slice * rank + min(rank, remainder)
        return weight[:, start:end]

      weights = [
          _slice_weight_for_rank(weight, info, self.rank)
          for weight, info in zip(weights, local_info)
      ]

    # now we have weight distributed, need to concat
    concat_weights = []
    for group in self.strategy.local_group_list[self.rank]:
      to_concat = [weights[idx] for idx in group]
      concat_weights.append(np.concatenate(to_concat))
    return concat_weights

  def set_row_slice_weights(self, weights):
    # we make sure no table is in this group in single worker case
    if not weights:
      return []
    if isinstance(weights[0], str):
      weights = [np.load(file=path, mmap_mode='r') for path in weights]
    local_info = [[1 for _ in range(self.world_size)] for _ in weights]

    def _slice_weight_for_rank(weight, info, global_rank):
      num_columns = weight.shape[0]
      num_slices = sum(info)
      column_per_slice = num_columns // num_slices
      remainder = num_columns % num_slices
      rank = sum(info[:global_rank])

      start = column_per_slice * rank + min(rank, remainder)
      rank += 1
      end = column_per_slice * rank + min(rank, remainder)
      return weight[start:end, :]

    weights = [
        _slice_weight_for_rank(weight, info, self.rank)
        for weight, info in zip(weights, local_info)
    ]
    return weights

  def set_weights(self, weights, chunk=134217728, use_lock=False):
    """Sets the weights of the layer, from NumPy arrays.

    Args:
      weights (list): list containing global weights for all table.
          item in the list can be either numpy array or file path to load from.
      chunk (int): max number of elements per chunk when set weight on GPU by chunks.
          this will be round to number of rows base on weight shape.
      use_lock (bool): If true, set weights rank by rank in lock step to avoid OOM. Default False.
    Raises:
      ValueError: If length of weights does not match length of expected weights.
    """
    if use_lock:
      for _ in range(self.rank):
        hvd.broadcast_object(0)

    dp_weights = [weights[idx] for idx in self.strategy.table_groups[0]]
    col_weights = [weights[idx] for idx in self.strategy.table_groups[1]]
    row_weights = [weights[idx] for idx in self.strategy.table_groups[2]]

    col_weights = self.set_col_slice_weights(col_weights)
    row_weights = self.set_row_slice_weights(row_weights)

    weights = dp_weights + col_weights + row_weights
    # variable.assign and copy-on-write creates extra copy of weight that causes OOM
    # so here we scatter update by ~128M elements chunks instead of just do
    # super().set_weights(weights)
    if len(self.weights) != len(weights):
      raise ValueError(
          F"You called `set_weights(weights)` on layer {self.name} with a weight list of "
          F"length {len(weights)}, but the layer was expecting {len(self.weights)} weights.")
    for weight, arr in zip(self.weights, weights):
      if arr.size <= chunk:
        weight.assign(arr)
      else:
        chunk_size_dim0 = chunk // weight.shape[1]
        num_chunks = math.ceil(weight.shape[0] / chunk_size_dim0)
        last_size = weight.shape[0] - chunk_size_dim0 * (num_chunks - 1)
        chunk_sizes = [chunk_size_dim0] * (num_chunks - 1) + [last_size]
        for i in range(num_chunks):
          start = i * chunk_size_dim0
          end = start + chunk_sizes[i]
          indices = tf.range(start=start, limit=end, dtype=tf.int64)
          update = tf.IndexedSlices(values=arr[start:end],
                                    indices=indices,
                                    dense_shape=weight.shape)
          weight.scatter_update(sparse_delta=update)

    del weights
    if use_lock:
      for _ in range(self.world_size - self.rank):
        hvd.broadcast_object(0)

  # 1d split that works beyond 32bit indexing limit TF support
  def _split_1d(self, tensor, lengths):
    # choose a number close to int32 limit as maximum chunk size
    # This will handle tensor with size up to square of int32_max
    chunking_threshold = 2147483646
    if tensor.shape[0] <= chunking_threshold:
      return tf.split(tensor, lengths)
    num_chunks = math.ceil(tensor.shape[0] / chunking_threshold)
    padding_len = math.ceil(tensor.shape[0] / num_chunks) * num_chunks - tensor.shape[0]
    padded_tensor = tf.concat([tensor, tf.zeros(padding_len, tensor.dtype)], axis=0)
    tensor_list = tf.unstack(tf.reshape(padded_tensor, [num_chunks, -1]))
    result = []
    for length in lengths:
      this_slice = []
      while length > 0:
        if length > tensor_list[0].shape[0]:
          this_slice.append(tensor_list.pop(0))
        else:
          this_slice.append(tensor_list[0][:length])
          tensor_list[0] = tensor_list[0][length:]
        length -= this_slice[-1].shape[0]
      result.append(tf.concat(this_slice, axis=0))
    return result

  def get_row_sliced_weights(self, weights):
    if not weights:
      return []
    # weights are already selected with group info
    # assume row slice run on all workers, then allgather conveniently stitch them together
    weights = hvd.grouped_allgather(weights)
    return [w.numpy() for w in weights]

  def get_col_sliced_weights(self, local_weights, all_ranks=False):
    if not local_weights:
      return []
    # TODO(deyuf): undo concat locally first. this require we save original local config
    if self.world_size == 1:
      concat_weights = [w.numpy() for w in local_weights]
      res = [item for sublist in self.strategy.local_group_list[0] for item in sublist]
      for offsets, f_w, group in zip(self.strategy.local_weight_offsets[0], concat_weights,
                                     self.strategy.local_group_list[0]):
        for i in range(len(offsets) - 1):
          res[group[i]] = f_w[offsets[i]:offsets[i + 1]]
      return res

    # mpi segfault on over 32bit range index, so we gather weights chunk by chunk here
    # choose a number not very close to int32 limit as maximum chunk size just to be safe
    chunking_threshold = 2000000000
    num_chunks = 1
    for local_config in self.strategy.local_configs:
      total_elements = sum([c['input_dim'] * c['output_dim'] for c in local_config])
      num_chunks = max(num_chunks, math.ceil(self.world_size * total_elements / chunking_threshold))

    with tf.device('CPU:0'):
      local_weights = tf.concat([tf.reshape(w, [-1]) for w in local_weights], axis=0)
      chunk_size = local_weights.shape[0] // num_chunks
      last_size = local_weights.shape[0] - chunk_size * (num_chunks - 1)
      chunk_sizes = [chunk_size] * (num_chunks - 1) + [last_size]
      local_weights = self._split_1d(local_weights, chunk_sizes)
      # communicate chunk sizes
      all_sizes = hvd.allgather(chunk_sizes)

      # collect all chunks and split to reverse allgather concat
      chunks = []
      for i, w in enumerate(local_weights):
        w = hvd.allgather(w)
        if all_ranks or self.rank == 0:
          chunks += self._split_1d(w, all_sizes[i::num_chunks])
      if not chunks:
        return []

      # re-construct all local weights from chunks
      local_weights = []
      for i in range(self.world_size):
        local_weights.append(tf.concat(chunks[i::self.world_size], axis=0))
      del chunks

      # split flat local weights into correct sizes
      weights = []
      for local_weight, local_config, weight_offsets, local_groups in zip(
          local_weights, self.strategy.local_configs, self.strategy.local_weight_offsets,
          self.strategy.local_group_list):
        local_shapes = [[c['input_dim'], c['output_dim']] for c in local_config]
        local_sizes = [shape[0] * shape[1] for shape in local_shapes]
        flat_weights = self._split_1d(local_weight, local_sizes)
        concat_weights = [
            tf.reshape(weight, shape) for weight, shape in zip(flat_weights, local_shapes)
        ]
        # split concat embedding weights
        res = [item for sublist in local_groups for item in sublist]
        for offsets, f_w, group in zip(weight_offsets, concat_weights, local_groups):
          for i in range(len(offsets) - 1):
            res[group[i]] = f_w[offsets[i]:offsets[i + 1]]
        weights += res

      # restore original table order
      # flatten self.strategy.table_ids
      worker_order_table_ids = [item for sublist in self.strategy.table_ids for item in sublist]
      # Shuffle worker ordered embedding weights(sliced) back to original order.
      ids_and_weights = sorted(zip(worker_order_table_ids, weights), key=lambda x: x[0])
      # concat sliced weights
      result = []
      cur_id = 0
      cur_list = []
      while ids_and_weights:
        cur = ids_and_weights.pop(0)
        if cur[0] == cur_id:
          cur_list.append(cur[1])
        else:
          result.append(tf.concat(cur_list, axis=1).numpy())
          cur_id = cur[0]
          cur_list = [cur[1]]
      result.append(tf.concat(cur_list, axis=1).numpy())
      return result

  def get_weights(self, all_ranks=False):
    """Returns the current weights of the layer, as NumPy arrays.

    This override outputs global weights for all tables.
    Args:
      all_ranks (bool): If true, return weights in all ranks, otherwise only in rank 0.
          Default False.
    Returns:
      result (list): List of weight tensors.
    """
    # avoid copy-on-read on dense access, assume order here for code simplicity
    weights = [read_var_no_copy(w) for w in self.weights]
    num_dp, num_col = len(self.dp_layers), len(self.local_embedding_layers)
    dp_weights = weights[:num_dp]
    col_weights = weights[num_dp:num_dp + num_col]
    row_weights = weights[num_dp + num_col:]

    col_weights = self.get_col_sliced_weights(col_weights, all_ranks)
    row_weights = self.get_row_sliced_weights(row_weights)

    weights = dp_weights + col_weights + row_weights
    group_order_table_ids = [idx for group in self.strategy.table_groups for idx in group]
    weights = [w for _, w in sorted(zip(group_order_table_ids, weights))]
    return weights

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if input_shape is not None and None not in input_shape[0]:
      # Do some checks to detect cases that are not supported
      if not isinstance(input_shape, (list, tuple)):
        input_shape = [input_shape]
      batch_sizes = [shape[0] for shape in input_shape]
      batch_sizes = hvd.allgather(batch_sizes).numpy().tolist()
      if len(set(batch_sizes)) > 1:
        raise ValueError(F"All input need to have same batchsize. got {set(batch_sizes)}.")
      if not self.dp_input:
        if batch_sizes[0] % self.world_size > 0:
          raise ValueError(
              F"Global batchsize {batch_sizes[0]} not divisible workers count {self.world_size}.")

    # build both col and row slice tables
    for layer in self.dp_layers:
      layer.build(input_shape[0] if input_shape else None)
      # set built flag to prevent above build trigger again and above flag fall off
      layer.built = True

    # build both col and row slice tables
    for layer in self.local_embedding_layers + self.row_layers:
      device = 'CPU:0' if layer.cpu_offloaded else 'GPU:0'
      with tf.device(device):
        layer.build(input_shape[0] if input_shape else None)
      for var in layer.trainable_weights:
        # Mark local(model parallel) variable. use prefix de(distributed embeddings) to avoid conflicts.
        var.de_local = True
      # set built flag to prevent above build trigger again and above flag fall off
      layer.built = True

    self.built = True

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    # call data parallel tables
    dp_in = [inputs[idx] for idx in self.strategy.input_groups[0]]
    dp_out = self._call_data_parallel(dp_in) if dp_in else []

    # call col slice tables
    col_in = [inputs[idx] for idx in self.strategy.input_groups[1]] if self.dp_input else inputs
    col_out = self._call_table_parallel(col_in) if col_in else []

    # call row slice tables
    row_in = [inputs[idx] for idx in self.strategy.input_groups[2]]
    row_out = self._call_row_slice(row_in) if row_in else []

    # now we have output from all groups, reorder them into input order
    outputs = dp_out + col_out + row_out
    outputs = [outputs[idx] for idx in self.strategy.rev_group_ids]
    return outputs


# Monkey patch horovod bcast/tape so we can handle mp/dp vars differently in single backward
# pylint: disable=protected-access, missing-any-param-doc, invalid-name
def broadcast_variables(model_vars, root_rank=0):
  """Broadcasts variables from root rank to all other processes in a process set

  Replace horovod's broadcast_variables when running hybrid parallel

  See https://horovod.readthedocs.io/en/stable/api.html for more details
  """
  dp_vars = []
  mp_vars = []
  for var in model_vars:
    if hasattr(var, 'de_local'):
      mp_vars.append(var)
    else:
      dp_vars.append(var)

  # modify broadcast to ignore_name_scope by default
  # TODO(deyuf): make it not positional
  _broadcast_defaults = list(hvd.broadcast.__defaults__)
  _broadcast_defaults[1] = True
  hvd.broadcast.__defaults__ = tuple(_broadcast_defaults)
  hvd.broadcast_variables(dp_vars, root_rank=root_rank)


def DistributedGradientTape(*args, **kwargs):
  """Graident tape that supports hybrid parallel

  Replace horovod's DistributedGradientTape when running hybrid parallel

  See https://horovod.readthedocs.io/en/stable/api.html for more details
  """

  def _gradient(self, target, sources, *args, **kwargs):
    # Overwrite use_generic_names to always be True
    kwargs["use_generic_names"] = True

    gradients = self.raw_gradient(target, sources, *args, **kwargs)
    return gradients

  if horovod.__version__ < '0.27.0':
    raise NotImplementedError(
        "DistributedGradientTape is only compatible with horovod 0.27 or newer.")
  tape = hvd.DistributedGradientTape(sparse_as_dense=True, *args, **kwargs)
  for var in tape.watched_variables():
    if hasattr(var, 'de_local'):
      tape.register_local_source(var)

  tape.raw_gradient = tape.gradient
  tape.gradient = types.MethodType(_gradient, tape)
  return tape


def DistributedOptimizer(*args, **kwargs):
  """Distributed optimizer that supports hybrid parallel

  Replace horovod's DistributedOptimizer when running hybrid parallel

  See https://horovod.readthedocs.io/en/stable/api.html for more details
  """

  # might be correct to patch get/aggregate gradient, but those seems already messy
  def _register_then_allreduce(self, grads, model_vars):
    if not self.local_var_registed:
      for var in model_vars:
        if hasattr(var, 'de_local'):
          self.register_local_var(var)
      self.local_var_registed = True
    return self.raw_allreduce(grads, model_vars)

  if horovod.__version__ < '0.27.0':
    raise NotImplementedError("Distributed Optimizer is only compatible with horovod 0.27 or newer")
  opt = hvd_keras.DistributedOptimizer(sparse_as_dense=True, *args, **kwargs)
  opt.local_var_registed = False
  opt.raw_allreduce = opt._allreduce
  opt._allreduce = types.MethodType(_register_then_allreduce, opt)

  # need to patch internal allreduce call with use_generic_names
  def _named_allreduce_grads(self, grads, variables):
    return self.raw_allreduce_grads(grads, variables, use_generic_names=True)

  opt.raw_allreduce_grads = opt._allreduce_grads
  opt._allreduce_grads = types.MethodType(_named_allreduce_grads, opt)
  return opt


def BroadcastGlobalVariablesCallback(*args, **kwargs):
  """Broadcast callback that supports hybrid parallel

  Replace horovod's BroadcastGlobalVariablesCallback when running hybrid parallel

  See https://horovod.readthedocs.io/en/stable/api.html for more details
  """

  def _on_batch_end(self, batch, logs=None):
    if not self.local_var_registed:
      for var in self.model.variables:
        if hasattr(var, 'de_local'):
          self.register_local_var(var)
      self.local_var_registed = True
    return self.raw_on_batch_end(batch, logs)

  if horovod.__version__ < '0.27.0':
    raise NotImplementedError(
        "BroadcastGlobalVariablesCallback is only compatible with horovod 0.27 or newer.")
  bcb = hvd_keras.callbacks.BroadcastGlobalVariablesCallback(*args, **kwargs)
  bcb.local_var_registed = False
  bcb.raw_on_batch_end = bcb.on_batch_end
  bcb.on_batch_end = types.MethodType(_on_batch_end, bcb)
  return bcb


# pylint: enable=protected-access, missing-any-param-doc, invalid-name
