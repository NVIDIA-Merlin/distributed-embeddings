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
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
import horovod.tensorflow as hvd
from .embedding import Embedding


class DistEmbeddingStrategy():
  """Distributed embedding strategy

  Args:
    embeddings (list of Embedding): list of unbuilt Embedding layers globally
    strategy (str): A string indicates how embedding tables are distributed.
        Choices are [“basic”, “memory_balanced”]. Default "basic"
    input_table_map (list or None): A list of table ids mapping each input to a table, i.e.,
        `input[i]` map to `table[input_table_map[i]]`. None means there are same number of
        inputs/tables and `input[i]` map to `table[i]`. Default None.
  """

  def __init__(self,
               embeddings,
               world_size,
               rank,
               strategy="basic",
               input_table_map=None,
               column_slice_threshold=None):
    self.global_configs = [e.get_config() for e in embeddings]
    self.strategy = strategy
    if input_table_map is None:
      input_table_map = list(range(len(embeddings)))
    if world_size == 1:
      self.local_configs = self.global_configs
      self.local_input_table_map = input_table_map
      self.input_ids_list = [list(range(len(input_table_map)))]
      self.table_ids_list = [list(range(len(embeddings)))]
      return
    # Create (maybe) sliced configs
    sliced_configs, self.sliced_out_ranges = self.create_sliced_configs(
        world_size, column_slice_threshold, input_table_map)
    # Apply strategy and save nested list containing table indices by rank
    self.table_ids_list = self.apply_stragety(strategy, world_size, sliced_configs)
    # Nested list to split embedding output from each rank into tables
    self.widths_list = []
    # Nested list containing input indices by rank
    self.input_ids_list = []
    # Nested list containing local input to local table map by rank
    self.local_map_list = []
    # Nested list containing local configs by rank
    self.local_configs_list = []
    # Each worker loop over all rank to get global view of strategy
    for rank_table_ids in self.table_ids_list:
      # calculate stats needed for each rank
      rank_widths, rank_input_ids, rank_input_map, rank_configs = [], [], [], []
      for m, table_idx in enumerate(rank_table_ids):
        rank_configs.append(sliced_configs[table_idx].pop(0))
        for k, mapped_idx in enumerate(input_table_map):
          if table_idx == mapped_idx:
            rank_widths.append(rank_configs[-1]['output_dim'])
            rank_input_ids.append(k)
            rank_input_map.append(m)
      self.local_configs_list.append(rank_configs)
      self.widths_list.append(rank_widths)
      self.input_ids_list.append(rank_input_ids)
      self.local_map_list.append(rank_input_map)
    # List of total embedding widths to split embedding output by rank after alltoall
    self.total_local_widths = [sum(widths) for widths in self.widths_list]
    # List that maps local inputs to local table
    self.local_input_table_map = self.local_map_list[rank]
    # flatten self.input_ids_list
    worker_order_input_ids = [item for sublist in self.input_ids_list for item in sublist]
    # List of indices to shuffle worker ordered embedding outputs back to original order
    self.rev_global_input_ids = [
        index
        for _, index in sorted(zip(worker_order_input_ids, range(len(worker_order_input_ids))))
    ]
    # List of configs to create local embedding layers
    self.local_configs = self.local_configs_list[rank]

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


class DistributedEmbedding(tf.keras.layers.Layer):
  """Distributed embedding wrapper

  This class is a hybrid parallel wrapper around embedding. It handles all to all communication of
  forward and backward of embedding.

  Args:
    embeddings (list of keras Embedding layers): embedding tables to be distributed
    strategy (str): A string indicates how embedding tables are distributed.
        Choices are [“basic”, “memory_balanced”]. Default "basic"
    column_slice_threshold (int or None): If not None, embedding tables with more elements than
        column_slice_threshold will be divide into N even pieces alone embedded width dimension.
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
    # TODO(Deyu): add more control over this with newly added hvd process_set api
    if not hvd.is_initialized():
      hvd.init()
    self.world_size = hvd.size()
    self.rank = hvd.rank()

    self.dp_input = dp_input
    self.column_slice_threshold = column_slice_threshold
    # get model parallel distribution strategy
    self.strategy = DistEmbeddingStrategy(embeddings,
                                          self.world_size,
                                          self.rank,
                                          strategy,
                                          input_table_map=input_table_map,
                                          column_slice_threshold=column_slice_threshold)
    if len(self.strategy.global_configs) < self.world_size:
      raise NotImplementedError

    # create local embeddings
    self.local_embedding_layers = []
    for config in self.strategy.local_configs:
      config['synchronization'] = tf.VariableSynchronization.NONE
      self.local_embedding_layers.append(Embedding.from_config(config))

  def _call_base(self, inputs):  # pylint: disable=missing-param-doc,missing-type-doc
    """Call function that do embeddings and communication

    Currently, it requires same batch_size on all workers.
    """
    # get model parallel input from data parallel
    if self.dp_input:
      comm_dtype = tf.int32
      for inp in inputs:
        if inp.dtype == tf.int64:
          comm_dtype = tf.int64
      inputs = [tf.cast(inp, comm_dtype) for inp in inputs]
      local_shapes, local_splits, global_splits, flat_inputs = [], [], [], []
      for rank_input_ids in self.strategy.input_ids_list:
        rank_inputs = [inputs[index] for index in rank_input_ids]
        local_shapes.append([inp.shape for inp in rank_inputs])
        rank_inputs = [tf.reshape(inp, [-1]) for inp in rank_inputs]
        local_splits.append([inp.shape[0] for inp in rank_inputs])
        global_splits.append(sum(local_splits[-1]))
        flat_inputs += rank_inputs
      inputs = tf.concat(flat_inputs, 0)
      inputs, _ = hvd.alltoall(inputs, splits=global_splits, name='inp_dp_to_mp')
      inputs = tf.reshape(inputs, [self.world_size, -1])
      inputs = tf.split(inputs, local_splits[self.rank], 1)
      inputs = [
          tf.reshape(inp, [self.world_size * shape[0]] + shape[1:])
          for inp, shape in zip(inputs, local_shapes[self.rank])
      ]

    # do embedding
    mp_outs = [
        self.local_embedding_layers[m](inp)
        for m, inp in zip(self.strategy.local_input_table_map, inputs)
    ]

    # concat last axis to make all2all slice correct, and reshape to make later split easier
    # TODO(Deyu): current assume 2D with same batch for all output, ideally should support general case
    local_bs = inputs[0].shape[0] // self.world_size
    mp_outs = tf.reshape(tf.concat(mp_outs, axis=-1), [-1, local_bs])
    dp_outs = hvd.alltoall(mp_outs, name='out_mp_to_dp')
    dp_outs = [
        tf.reshape(t, [local_bs, -1]) for t in tf.split(dp_outs, self.strategy.total_local_widths)
    ]
    # split each worker result and re-order using id
    worker_order_res = []
    for dp_out, widths in zip(dp_outs, self.strategy.widths_list):
      worker_order_res += tf.split(dp_out, widths, 1)
    # reorder outputs to be same as inputs order
    result = [worker_order_res[index] for index in self.strategy.rev_global_input_ids]
    return result

  def _concat_column_slice_outputs(self, outs):
    """Concat sliced outputs result from column slicing back together
    """
    for start, end in self.strategy.sliced_out_ranges:
      outs[start:end] = [tf.concat(outs[start:end], axis=-1)]
    return outs

  def set_weights(self, weights):  # pylint: disable=missing-param-doc,missing-type-doc
    """Sets the weights of the layer, from NumPy arrays.

    This override expects global weights for all tables as input.
    """
    if self.world_size == 1:
      sliced_local_weights = weights
    else:
      slice_info = [[rank_tids.count(tid)
                     for rank_tids in self.strategy.table_ids_list]
                    for tid in range(len(weights))]
      local_weights = [weights[index] for index in self.strategy.table_ids_list[self.rank]]
      local_info = [slice_info[index] for index in self.strategy.table_ids_list[self.rank]]

      def _slice_weight_for_rank(weight, info, global_rank):
        num_columns = weight.shape[1]
        num_slices = sum(info)
        column_per_slice = num_columns // num_slices
        remainder = num_columns % num_slices
        rank = info[:global_rank].count(1)

        start = column_per_slice * rank + min(rank, remainder)
        rank += 1
        end = column_per_slice * rank + min(rank, remainder)
        return weight[:, start:end]

      sliced_local_weights = [
          _slice_weight_for_rank(weight, info, self.rank)
          for weight, info in zip(local_weights, local_info)
      ]
    super().set_weights(sliced_local_weights)

  def get_weights(self):
    """Returns the current weights of the layer, as NumPy arrays.

    This override outputs global weights for all tables.
    """
    if self.world_size == 1:
      return [weight.numpy() for weight in self.weights]

    # mpi segfault on large sizes so we gather weights chunk by chunk here
    num_chunks = 8
    with tf.device('CPU:0'):
      local_weights = tf.concat([tf.reshape(w, [-1]) for w in self.weights], axis=0)
      chunk_size = local_weights.shape[0] // num_chunks
      last_size = local_weights.shape[0] - chunk_size * (num_chunks - 1)
      chunk_sizes = [chunk_size] * (num_chunks - 1) + [last_size]
      local_weights = tf.split(local_weights, chunk_sizes)
      all_sizes = hvd.allgather(chunk_sizes)

      # collect all chunks and split to reverse allgather concat
      chunks = []
      for i, w in enumerate(local_weights):
        chunks += tf.split(hvd.allgather(w), all_sizes[i::num_chunks])
      # re-construct all local weights from chunks
      local_weights = []
      for i in range(self.world_size):
        local_weights.append(tf.concat(chunks[i::self.world_size], axis=0))
      # split flat local weights into correct sizes
      weights = []
      for local_weight, local_configs in zip(local_weights, self.strategy.local_configs_list):
        local_shapes = [[c['input_dim'], c['output_dim']] for c in local_configs]
        local_sizes = [shape[0] * shape[1] for shape in local_shapes]
        flat_weights = tf.split(local_weight, local_sizes)
        weights += [tf.reshape(weight, shape) for weight, shape in zip(flat_weights, local_shapes)]
      # restore original table order
      # flatten self.strategy.table_ids_list
      worker_order_table_ids = [
          item for sublist in self.strategy.table_ids_list for item in sublist
      ]
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
    for layer in self.local_embedding_layers:
      layer.build(input_shape)
    self.built = True

  def call(self, inputs):  # pylint: disable=missing-function-docstring
    if self.world_size == 1:
      outputs = [
          self.local_embedding_layers[m](inp)
          for m, inp in zip(self.strategy.local_input_table_map, inputs)
      ]
      return outputs

    # TODO(skyw): Revisit logics of selecting call functions for different strategy
    outputs = self._call_base(inputs)
    outputs = self._concat_column_slice_outputs(outputs)
    return outputs


# Monkey patch horovod bcast/tape so we can handle mp/dp vars differently in single backward
def broadcast_variables(model_vars, root_rank=0):  # pylint: disable=missing-any-param-doc
  """Broadcasts variables from root rank to all other processes in a process set

  Replace horovod's broadcast_variables when running hybrid parallel

  See https://horovod.readthedocs.io/en/stable/api.html for more details
  """
  dp_vars = []
  mp_vars = []
  for var in model_vars:
    if var.synchronization == tf.VariableSynchronization.NONE:
      mp_vars.append(var)
    else:
      dp_vars.append(var)
  hvd.broadcast_variables(dp_vars, root_rank=root_rank)


def DistributedGradientTape(*args, **kwargs):  # pylint: disable=missing-param-doc,invalid-name
  """Graident tape that supports hybrid parallel

  Replace horovod's DistributedGradientTape when running hybrid parallel

  See https://horovod.readthedocs.io/en/stable/api.html for more details
  """

  def gradient(self, target, sources, output_gradients=None):
    gradients = super(self.__class__, self).gradient(target, sources, output_gradients)
    dp_vars = []
    dp_grads = []
    mp_grads = []
    split_infos = []
    for grad, var in zip(gradients, sources):
      if var.synchronization == tf.VariableSynchronization.NONE:
        if isinstance(grad, tf.IndexedSlices):
          mp_grads.append(tf.IndexedSlices(grad.values / hvd.size(), grad.indices,
                                           grad.dense_shape))
        else:
          mp_grads.append(grad / hvd.size())
        split_infos.append((True, len(mp_grads) - 1))
      else:
        dp_vars.append(var)
        dp_grads.append(grad)
        split_infos.append((False, len(dp_grads) - 1))
    dp_grads = self._allreduce_grads(dp_grads, dp_vars)  # pylint: disable=protected-access
    # put gradients back in original order
    grads = []
    for info in split_infos:
      if info[0]:
        grads.append(mp_grads[info[1]])
      else:
        grads.append(dp_grads[info[1]])
    return grads

  tape = hvd.DistributedGradientTape(*args, **kwargs)
  setattr(type(tape), 'gradient', gradient)
  return tape
