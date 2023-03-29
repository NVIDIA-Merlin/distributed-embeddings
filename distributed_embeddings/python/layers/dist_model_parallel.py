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

  def __call__(self, shape, **kwargs):
    weights = [self._initializer([size, shape[1]], **kwargs) for size in self.sizes]
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


class DistEmbeddingStrategy():
  """Distributed embedding strategy

  Args:
    embeddings (list of Embedding): list of unbuilt Embedding layers globally
    strategy (str): A string indicates how embedding tables are distributed.
        Choices are [“basic”, “memory_balanced”]. Default "basic"
    input_table_map (list or None): A list of table ids mapping each input to a table, i.e.,
        `input[i]` map to `table[input_table_map[i]]`. None means there are same number of
        inputs/tables and `input[i]` map to `table[i]`. Default None.

  Attributes:
    strategy: string indicates how embedding tables are distributed.
    column_slice_threshold: desired upper bound of elements count in each slice.
    sliced_out_ranges: list of [output_pos, num_slices] for each input, used to merged outputs.
    input_ids_list: nested list, contain input index list in rank order. use for dp_input == False.
    local_maps: nested list, contain per rank local input to table map.
    local_configs: nested list, contain per rank lists of local table configs.
    local_input_offsets: nested list, contain per rank offsets to form input for concat embedding.
    local_weight_offsets: nested list, contain per rank weight internal offsets get/set_weights.
    local_group_list: nested list, contain per rank concat grouping info for get/set_weights.
    table_ids: nested list, contain per rank table ids for get/set_weights.
    widths_list_flat: list of all output width, before merging slices and in worker order
    rev_global_input_ids: list to shuffle output in worker order into same input order.
  """

  def __init__(self,
               embeddings,
               world_size,
               strategy="basic",
               input_table_map=None,
               column_slice_threshold=None):
    # code in DMP to skip hvd call in single process case may assume "basic"
    self.strategy = "basic" if world_size == 1 else strategy
    # column_slice can be used to enable more table concat, so keep it in single process
    self.column_slice_threshold = column_slice_threshold
    self.global_configs = [e.get_config() for e in embeddings]
    # Insert layer type information to config dicts
    for config, embedding in zip(self.global_configs, embeddings):
      config['layer_type'] = type(embedding)
    if input_table_map is None:
      input_table_map = list(range(len(embeddings)))

    # Create (maybe) sliced configs
    sliced_configs, self.sliced_out_ranges = self.create_sliced_configs(
        world_size, self.column_slice_threshold, input_table_map)

    # Apply strategy and save nested list containing table indices by rank
    table_ids = self.apply_stragety(self.strategy, world_size, sliced_configs)

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
        for k, mapped_idx in enumerate(input_table_map):
          if table_idx == mapped_idx:
            rank_input_ids.append(k)
            rank_input_map.append(m)

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
    # TODO(deyuf): might switch to not use this to support non-2D output
    self.widths_list_flat = []
    for config, input_map in zip(self.local_configs, self.local_maps):
      self.widths_list_flat += [config[m]['output_dim'] for m in input_map]

    # List of indices to shuffle worker ordered embedding outputs back to original order
    worker_order_input_ids = [item for sublist in self.input_ids_list for item in sublist]
    self.rev_global_input_ids = [
        index
        for _, index in sorted(zip(worker_order_input_ids, range(len(worker_order_input_ids))))
    ]

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

  def create_sliced_configs(self, world_size, column_slice_threshold, input_table_map):
    """Create column sliced configs from global configs.
    This function also calculate ranges of data parallel output needs concat due to this slice.
    Args:
      world_size (int): number of model parallel workers
      column_slice_threshold (int or None): desired upper bound of elements count in each slice
      input_table_map (list): A list of table ids mapping each input to a table
    Returns:
      sliced_configs (list): same length as global configs. each element is a list represent sliced
    form of global config at the same position.
      sliced_out_ranges (list): each element is list of 2 integers, representing output ranges need
    to be concatenated to re-form output due to above slice.
    """
    # TODO(Deyu): in auto slice and when there are equal sized tables, allow slice some of them
    # less table than worker, we try our best to slice into worker count slices(may go over)
    if column_slice_threshold is None:
      table_sizes = [config['input_dim'] * config['output_dim'] for config in self.global_configs]
      while world_size > len(table_sizes):
        table_sizes.sort()
        column_slice_threshold = table_sizes[-1] - 1
        cur_max_size = table_sizes.pop(-1)
        table_sizes += [cur_max_size // 2, cur_max_size // 2]

    sliced_configs = []
    for global_config in self.global_configs:
      maybe_sliced_config = self.maybe_slice_table_column(global_config, column_slice_threshold,
                                                          world_size)
      sliced_configs.append(maybe_sliced_config)
    # figure out ranges of output that needs concat
    # this needs to be in output order, otherwise range modification would fail
    sliced_out_ranges = []
    for input_id, table_id in enumerate(input_table_map):
      if len(sliced_configs[table_id]) > 1:
        sliced_out_ranges.append([input_id, input_id + len(sliced_configs[table_id])])
    return sliced_configs, sliced_out_ranges

  # pylint: disable=missing-param-doc,missing-type-doc,missing-raises-doc
  def apply_stragety(self, mode, world_size, sliced_configs):
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

  # First attempt here, converting into shared embedding and let TF/XLA does the job.
  # TODO(deyuf): add shortcut for all to 1 concat. explicitly shape things to save slice/concat
  # i.e. reshape to [num_worker, num_inp, local_batch] and offset in [1, num_inp, 1]
  def _create_concat(self, table_configs, input_maps):
    # first get local table id into groups
    grouped_table_ids, concat_configs = [], []
    for table_id, config in enumerate(table_configs):
      for group, concat_config in zip(grouped_table_ids, concat_configs):
        if config['output_dim'] == concat_config['output_dim'] and config.get(
            'combiner') == concat_config.get('combiner'):
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
        # TODO(deyuf): custom layer without initializer will be concat but init is not wrapped
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
    row_slice (TBD): Describe how which embedding needs to be row sliced
    dp_input (bool): If True, takes data parallel input, i.e. in shape
        [local_batch_size x global_num_embeddings]. Otherwise take model parall input in shape
        [global_batch_size x local_num_embeddings]. Default True.
    input_table_map (list or None): same length list as inputs, map `input[i]`
        to `table[input_table_map[i]]`. None means there are same number of
        inputs/tables and `input[i]` map to `table[i]`. Default None.
  """

  def __init__(self,
               embeddings,
               strategy="basic",
               column_slice_threshold=None,
               row_slice=None,
               dp_input=True,
               input_table_map=None,
               **kwargs):

    super().__init__(**kwargs)
    if strategy not in ['basic', 'memory_balanced', 'memory_optimized']:
      raise ValueError(F"Unsupported shard strategy {strategy}")
    if row_slice is not None:
      raise NotImplementedError("Row slicing embedding is not supported yet!")

    # Currently assume data parallel ranks == model parallel ranks
    # TODO(deyuf): add more control over this with newly added hvd process_set api
    if not hvd.is_initialized():
      hvd.init()
    self.world_size = hvd.size()
    self.rank = hvd.rank()

    self.dp_input = dp_input
    self.column_slice_threshold = column_slice_threshold
    # get model parallel distribution strategy
    self.strategy = DistEmbeddingStrategy(embeddings,
                                          self.world_size,
                                          strategy,
                                          input_table_map=input_table_map,
                                          column_slice_threshold=column_slice_threshold)
    # Handle explicit threshold or corner cases, in which worker may receive no configs
    if not all(rank_configs for rank_configs in self.strategy.local_configs):
      raise ValueError("Not enough table after slicing to run on all worker."
                       "Try decrease column_slice_threshold or decrease worker count")

    # create local embeddings
    self.local_embedding_layers = []
    for config in self.strategy.local_configs[self.rank]:
      layer_type = config.pop('layer_type')
      # For stock keras Embedding, we switch underlying layer for better performance
      # If inputs are custom layers, original layer will be used
      # TODO(deyuf): Check functionality coverage, add fallback or type picking api
      layer_type = Embedding if layer_type == tf.keras.layers.Embedding else layer_type
      self.local_embedding_layers.append(layer_type.from_config(config))
    self.offsets = [
        None if offset == 0 else tf.constant([offset], dtype=tf.int64)
        for offset in self.strategy.local_input_offsets[self.rank]
    ]

  def _call_base(self, inputs):  # pylint: disable=missing-param-doc,missing-type-doc
    """Call function that do embeddings and communication

    Currently, it requires same batch_size on all workers.
    """
    # get model parallel input from data parallel
    if self.dp_input:
      if self.world_size > 1:
        comm_dtype = tf.int32
        for inp in inputs:
          if inp.dtype == tf.int64:
            comm_dtype = tf.int64
        inputs = [tf.cast(inp, comm_dtype) for inp in inputs]
        local_shapes, local_splits, global_splits, flat_inputs = [], [], [], []
        for rank_input_ids in self.strategy.input_ids_list:
          rank_inputs = [inputs[index] for index in rank_input_ids]
          local_shapes.append([_get_shape(inp) for inp in rank_inputs])
          rank_inputs = [tf.reshape(inp, [-1]) for inp in rank_inputs]
          local_splits.append([_get_shape(inp)[0] for inp in rank_inputs])
          global_splits.append(sum(local_splits[-1]))
          flat_inputs += rank_inputs
        inputs = tf.concat(flat_inputs, 0)
        inputs, _ = hvd.alltoall(inputs, splits=global_splits, name='inp_dp_to_mp')
        inputs = tf.reshape(inputs, [self.world_size, -1])
        inputs = tf.split(inputs, local_splits[self.rank], 1)
        inputs = [
            tf.reshape(inp, tf.concat([[self.world_size * shape[0]], shape[1:]], 0))
            for inp, shape in zip(inputs, local_shapes[self.rank])
        ]
      else:
        # expected input order may still change in case of single process
        inputs = [inputs[idx] for idx in self.strategy.input_ids_list[0]]

    if len(inputs) != len(self.strategy.local_maps[self.rank]):
      raise ValueError(F"Expect {self.strategy.local_maps[self.rank]} inputs, got {len(inputs)}.")

    # offset inputs
    inputs = [
        inp if offset is None else inp + tf.cast(offset, inp.dtype)
        for inp, offset in zip(inputs, self.offsets)
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
    result = [mp_outs[index] for index in self.strategy.rev_global_input_ids]
    return result

  def _concat_column_slice_outputs(self, outs):
    """Concat sliced outputs result from column slicing back together"""
    for start, end in self.strategy.sliced_out_ranges:
      outs[start:end] = [tf.concat(outs[start:end], axis=-1)]
    return outs

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

    if self.world_size > 1:
      slice_info = [[rank_table_id.count(table_id)
                     for rank_table_id in self.strategy.table_ids]
                    for table_id in range(len(weights))]
      weights = [weights[index] for index in self.strategy.table_ids[self.rank]]
      if isinstance(weights[0], str):
        weights = [np.load(file=path, mmap_mode='r') for path in weights]
      local_info = [slice_info[index] for index in self.strategy.table_ids[self.rank]]

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
    else:
      if isinstance(weights[0], str):
        weights = [np.load(file=path, mmap_mode='r') for path in weights]
    # now we have weight distributed, need to concat
    concat_weights = []
    for group in self.strategy.local_group_list[self.rank]:
      to_concat = [weights[idx] for idx in group]
      concat_weights.append(np.concatenate(to_concat))
    weights = concat_weights
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

  def get_weights(self, all_ranks=False):
    """Returns the current weights of the layer, as NumPy arrays.

    This override outputs global weights for all tables.
    Args:
      all_ranks (bool): If true, return weights in all ranks, otherwise only in rank 0.
          Default False.
    Returns:
      result (list): List of weight tensors.
    """
    # avoid copy-on-read on dense access
    local_weights = [read_var_no_copy(w) for w in self.weights]
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

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    if input_shape is not None:
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
    for layer in self.local_embedding_layers:
      layer.build(input_shape[0] if input_shape else None)
      for var in layer.trainable_weights:
        # Mark local(model parallel) variable. use prefix de(distributed embeddings) to avoid conflicts.
        var.de_local = True
      # set built flag to prevent above build trigger again and above flag fall off
      layer.built = True
    self.built = True

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    # TODO(skyw): Revisit logics of selecting call functions for different strategy
    outputs = self._call_base(inputs)
    outputs = self._concat_column_slice_outputs(outputs)
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
  tape = hvd.DistributedGradientTape(*args, **kwargs)
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
  opt = hvd_keras.DistributedOptimizer(*args, **kwargs)
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
