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
"""Test of distributed model parallel"""
import random
import os
from collections import defaultdict

from absl import flags
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.keras import keras_parameterized
import horovod.tensorflow as hvd

from distributed_embeddings.python.layers import dist_model_parallel as dmp
from distributed_embeddings.python.layers.embedding import Embedding
from distributed_embeddings.python.layers.dist_model_parallel import _dp_to_mp_input

flags.DEFINE_integer("seed", default=42, help="Random seed for the randomized tests.")
flags.DEFINE_bool("graph_mode", default=False, help="Run in graph mode.")
flags.DEFINE_string("mixed_precision_policy",
                    default=None,
                    help="Mixed precision policy to be set.")

FLAGS = flags.FLAGS

large_testcase_sizes = [[2, 8], [2, 16], [10, 8], [10, 16], [10, 16], [10, 16], [10, 16], [10, 16],
                        [10, 32], [10, 128], [10, 128], [10, 128], [10, 128], [10, 1024], [100, 16],
                        [100, 32], [100, 32], [100, 32], [100, 32], [100, 128], [100, 128],
                        [1000, 16], [1000, 16], [1000, 48], [1000, 128], [1000, 128], [1000, 384],
                        [10000, 64], [10000, 64], [10000, 2048], [100000, 32], [100000, 64],
                        [100000, 64], [100000, 64], [100000, 128], [1000000, 96], [1000000, 128],
                        [1000000, 128], [9999999, 8], [10000000, 8], [10000001, 8]]


# There are some functions in TF that pylint can't inspect correctly which leads to incorrect
# report of unexpected-keyword-arg, no-value-for-parameter. Disable them globally here
# pylint: disable=no-self-use,unexpected-keyword-arg,no-value-for-parameter,missing-docstring
class CustomEmbedding(tf.keras.layers.Layer):

  def __init__(self, input_dim, output_dim, **kwargs):
    super().__init__(**kwargs)
    self.input_dim = input_dim
    self.output_dim = output_dim

  def build(self, _):
    self.params = self.add_weight("params",
                                  shape=[self.input_dim, self.output_dim],
                                  dtype=tf.float32)

  def call(self, inputs):
    return tf.gather(params=self.params, indices=inputs, axis=None)

  def get_config(self):
    config = {'input_dim': self.input_dim, 'output_dim': self.output_dim}
    return config


class EmbeddingListModel(tf.keras.Model):
  """A simple model for test"""

  def __init__(self,
               table_sizes,
               distribute=False,
               strategy='basic',
               dp_input=True,
               input_table_map=None,
               column_slice_threshold=None,
               test_custom_layer=False,
               combiner=None,
               use_custom_kernels=True,
               row_slice_threshold=None,
               data_parallel_threshold=None,
               gpu_embedding_size=None):
    super().__init__()
    self.embeddings = []
    for size in table_sizes:
      if test_custom_layer:
        self.embeddings.append(CustomEmbedding(*size))
      elif combiner is None:
        self.embeddings.append(tf.keras.layers.Embedding(*size))
      else:
        self.embeddings.append(
            Embedding(*size, combiner=combiner, use_custom_kernel=use_custom_kernels))
    if distribute:
      self.dist_embeddings = dmp.DistributedEmbedding(
          self.embeddings,
          strategy=strategy,
          dp_input=dp_input,
          input_table_map=input_table_map,
          column_slice_threshold=column_slice_threshold,
          row_slice_threshold=row_slice_threshold,
          data_parallel_threshold=data_parallel_threshold,
          gpu_embedding_size=gpu_embedding_size)
    else:
      self.dist_embeddings = None
    self.dense = tf.keras.layers.Dense(5)
    self.input_table_map = input_table_map

  @tf.function
  def call(self, inputs):
    if self.dist_embeddings is not None:
      outs = self.dist_embeddings(inputs)
    elif self.input_table_map:
      outs = [self.embeddings[j](i) for i, j in zip(inputs, self.input_table_map)]
    else:
      outs = [e(i) for i, e in zip(inputs, self.embeddings)]
    out = tf.concat(outs, 1)
    return self.dense(out)

  def get_config(self):
    """
    get_config is an abstrct method in keras.Model, which implies it is important to be explicit
    although we don't use it in the test
    """
    return None


def initialize_hvd():
  os.environ['HOROVOD_STALL_CHECK_TIME_SECONDS'] = '5'
  os.environ['HOROVOD_STALL_SHUTDOWN_TIME_SECONDS'] = '30'

  hvd.init()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


def gen_table_sizes(num_tables=None):
  random.seed(FLAGS.seed)
  if num_tables is None:
    num_tables = random.randint(1, 2 * hvd.size())
  table_sizes = []
  for _ in range(num_tables):
    table_height = random.randint(4, 20)
    table_width = random.randint(4, 15)
    table_sizes.append([table_height, table_width])
  return table_sizes


def gen_input_to_table_map(num_tables):
  # make sure each table have input
  mapping = list(range(num_tables))
  # fix number of shared input to 3
  for _ in range(3):
    mapping.append(random.randint(0, num_tables - 1))
  random.shuffle(mapping)
  return mapping


def gen_inputs(*args, hotness=None, **kwargs):
  if hotness is None:
    return gen_inputs_onehot(*args, **kwargs)

  return gen_inputs_multihot(*args, hotness=hotness, **kwargs)


def gen_inputs_onehot(global_batch,
                      table_sizes,
                      input_to_table_map=None,
                      mp_input_ids=None,
                      return_global=False):
  # create global inputs
  if input_to_table_map is None:
    input_to_table_map = list(range(len(table_sizes)))
  global_inputs = [
      tf.random.uniform(shape=[global_batch], minval=0, maxval=table_sizes[i][0], dtype=tf.int64)
      for i in input_to_table_map
  ]
  for t in global_inputs:
    hvd.broadcast(t, root_rank=0)

  if return_global:
    return global_inputs

  local_batch = global_batch // hvd.size()

  dp_inputs = [t[hvd.rank() * local_batch:(hvd.rank() + 1) * local_batch] for t in global_inputs]
  mp_inputs = [global_inputs[i] for i in mp_input_ids] if mp_input_ids else []

  return dp_inputs, mp_inputs


def gen_inputs_multihot(global_batch,
                        table_sizes,
                        hotness=None,
                        input_to_table_map=None,
                        mp_input_ids=None,
                        return_global=False):
  # create global inputs
  if input_to_table_map is None:
    input_to_table_map = list(range(len(table_sizes)))

  if isinstance(hotness, int):
    hotness = [hotness for _ in range(len(input_to_table_map))]

  global_inputs = []
  for i, hotness_i in zip(input_to_table_map, hotness):
    t = tf.random.uniform(shape=[global_batch * hotness_i],
                          minval=0,
                          maxval=table_sizes[i][0],
                          dtype=tf.int64)
    hvd.broadcast(t, root_rank=0)
    row_lengths = tf.ones(shape=[global_batch], dtype=tf.int64) * hotness_i
    t = tf.RaggedTensor.from_row_lengths(values=t, row_lengths=row_lengths)
    global_inputs.append(t)

  if return_global:
    return global_inputs

  local_batch = global_batch // hvd.size()
  dp_inputs = [t[hvd.rank() * local_batch:(hvd.rank() + 1) * local_batch, :] for t in global_inputs]
  mp_inputs = [global_inputs[i] for i in mp_input_ids] if mp_input_ids else []

  return dp_inputs, mp_inputs


class DistributedEmbeddingTest(keras_parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    initialize_hvd()
    self.global_batch = 24

    tf.config.run_functions_eagerly(not flags.FLAGS.graph_mode)

    if flags.FLAGS.mixed_precision_policy:
      policy = tf.keras.mixed_precision.Policy(flags.FLAGS.mixed_precision_policy)
      tf.keras.mixed_precision.set_global_policy(policy)

  def run_and_test(self,
                   ref_model,
                   ref_inputs,
                   test_model,
                   test_inputs,
                   fwd_tol=None,
                   bwd_tol=1e-5):
    tf.keras.utils.set_random_seed(hvd.rank())
    # run a batch to initialize weight tensors
    _ = ref_model(ref_inputs)
    _ = test_model(test_inputs)

    # broadcast ref model weights and set test model weights
    hvd.broadcast_variables(ref_model.variables, root_rank=0)
    ref_weights = ref_model.get_weights()
    num_tables = len(ref_model.embeddings)
    test_model.dist_embeddings.set_weights(ref_weights[:num_tables])
    test_model.dense.set_weights(ref_weights[num_tables:])

    with tf.GradientTape() as tape:
      ref_out = tf.cumsum(ref_model(ref_inputs), axis=1)
    tape = hvd.DistributedGradientTape(tape)
    ref_grads = tape.gradient(ref_out, ref_model.variables)

    with tf.GradientTape() as tape:
      test_out = tf.cumsum(test_model(test_inputs), axis=1)
    tape = dmp.DistributedGradientTape(tape)
    test_grads = tape.gradient(test_out, test_model.variables)

    # assert forward result match
    if fwd_tol is None:
      self.assertAllEqual(ref_out, test_out)
    else:
      self.assertAllClose(ref_out, test_out, rtol=fwd_tol, atol=fwd_tol)

    # slicing grad is tricky. so we check weights updated with grad
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.5, momentum=0)
    optimizer.apply_gradients(zip(ref_grads, ref_model.variables))
    optimizer.apply_gradients(zip(test_grads, test_model.variables))
    ref_weights = ref_model.get_weights()
    test_weights = test_model.dist_embeddings.get_weights(True) + test_model.dense.get_weights()

    for ref_w, test_w in zip(ref_weights, test_weights):
      # assert close here since order of accumulations(inputs and batch dim) might have changed
      self.assertAllClose(tf.convert_to_tensor(ref_w),
                          tf.convert_to_tensor(test_w),
                          rtol=bwd_tol,
                          atol=bwd_tol)

  def test_broadcast(self):
    tf.keras.utils.set_random_seed(hvd.rank())
    num_tables = 7
    table_sizes = [[11, 7], [5, 8], [3, 8], [5, 8], [12, 25], [3, 12], [7, 13]]

    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')
    # create same input on all worker
    dup_ids = [
        tf.random.uniform(shape=[3], minval=0, maxval=table_sizes[i][0], dtype=tf.int64)
        for i in range(num_tables)
    ]
    dup_ids = hvd.broadcast_object(dup_ids, root_rank=0)
    # different output from each worker with different weight
    test_out = test_model(dup_ids)
    test_outs = tf.unstack(hvd.allgather(tf.expand_dims(test_out, axis=0)))
    for idx, out1 in enumerate(test_outs):
      for out2 in test_outs[idx + 1:]:
        self.assertNotAllClose(out1, out2)

    # same output from each worker after broadcast data parallel weights
    dmp.broadcast_variables(test_model.variables, root_rank=0)
    test_out = test_model(dup_ids)
    test_outs = tf.unstack(hvd.allgather(tf.expand_dims(test_out, axis=0)))
    for out1 in test_outs[1:]:
      self.assertAllEqual(test_outs[0], out1)

  def test_basic(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_row_slice(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    row_slice_threshold=1)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_data_parallel(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    data_parallel_threshold=100000)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_memory_optimized(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='memory_optimized')

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_shared_basic(self):
    table_sizes = gen_table_sizes()
    input_to_table_map = gen_input_to_table_map(len(table_sizes))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   input_table_map=input_to_table_map)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    input_table_map=input_to_table_map)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes, input_to_table_map)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_shared_basic_mp(self):
    table_sizes = gen_table_sizes()
    input_to_table_map = gen_input_to_table_map(len(table_sizes))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   input_table_map=input_to_table_map)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    dp_input=False,
                                    input_table_map=input_to_table_map)

    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]
    dp_inputs, mp_inputs = gen_inputs(self.global_batch, table_sizes, input_to_table_map,
                                      mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_shared_mb_mp(self):
    table_sizes = gen_table_sizes()
    input_to_table_map = gen_input_to_table_map(len(table_sizes))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   input_table_map=input_to_table_map)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    dp_input=False,
                                    input_table_map=input_to_table_map)

    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]
    dp_inputs, mp_inputs = gen_inputs(self.global_batch, table_sizes, input_to_table_map,
                                      mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_column_slice_merge(self):
    # test on 4 GPUs
    table_sizes = [[100, 8], [5, 8], [10, 8], [25, 4]]
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    column_slice_threshold=45)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)
    for tables in test_model.dist_embeddings.strategy.table_ids:
      self.assertEqual(len(tables), len(set(tables)))

  def test_column_slice_threshold(self):
    table_sizes = gen_table_sizes(hvd.size() + 1)
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    column_slice_threshold=30)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_column_slice_dup_worker(self):
    table_sizes = [[10, 4], [11, 2], [4, 2], [4, 2]]
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    dp_input=False,
                                    column_slice_threshold=10)
    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]
    dp_inputs, mp_inputs = gen_inputs(self.global_batch, table_sizes, mp_input_ids=mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_8table_width2_auto_concat(self):
    table_sizes = [[10, 2], [11, 2], [4, 2], [4, 2], [10, 2], [11, 2], [4, 2], [4, 2]]
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    dp_input=False)
    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]
    dp_inputs, mp_inputs = gen_inputs(self.global_batch, table_sizes, mp_input_ids=mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)
    self.assertEqual(len(test_model.dist_embeddings.weights), 1, "Table fusion failed.")

  def test_set_weight_uninitialized(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)

    # run a batch to initialize weight tensors
    _ = ref_model(dp_inputs)
    ref_weights = ref_model.get_weights()
    num_tables = len(ref_model.embeddings)
    with self.assertRaises(ValueError):
      test_model.dist_embeddings.set_weights(ref_weights[:num_tables])
      test_model.dense.set_weights(ref_weights[num_tables:])

  def test_indivisible_batch(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic', dp_input=False)

    # First generate model parallel batches that's divisible by world_size. We then use (batch_size - 1)
    # which will be indivisible by world_size greater than 1 due to consecutive numbers coprimes
    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]
    dp_inputs, mp_inputs = gen_inputs(self.global_batch, table_sizes, mp_input_ids=mp_input_ids)
    mp_inputs = [inp[1:] for inp in mp_inputs]
    if hvd.size() > 1:
      with self.assertRaisesRegex(ValueError, "not divisible"):
        self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_fewer_tables_than_workers(self):
    table_sizes = gen_table_sizes(1)

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='memory_balanced')

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_custom_embedding_layer(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False, test_custom_layer=True)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    test_custom_layer=True)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_all_parallelism_modes(self):
    """
    This testcase is designed to check if all parallelism modes,
    i.e., data-parallelism, table-parallism, column-parallelism,
    and row-parallelism, work together correctly.
    """

    table_sizes = large_testcase_sizes

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    data_parallel_threshold=10000,
                                    column_slice_threshold=1000000,
                                    row_slice_threshold=10000000)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_cpu_offload(self):
    table_sizes = [[100, 32], [100, 32], [100, 32], [100, 32], [1000, 64], [1000, 64], [1000, 64],
                   [1000, 64]]
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    gpu_embedding_size=32000)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs, fwd_tol=1e-6)

  def test_column_slicing_offload(self):
    table_sizes = large_testcase_sizes

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    column_slice_threshold=1000000,
                                    gpu_embedding_size=0)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs, fwd_tol=1e-6)

  def test_multihot_mp_input(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   combiner='sum',
                                   use_custom_kernels=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    combiner='sum',
                                    strategy='basic',
                                    dp_input=False)

    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]

    dp_inputs, mp_inputs = gen_inputs(self.global_batch,
                                      table_sizes,
                                      mp_input_ids=mp_input_ids,
                                      hotness=5)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs, fwd_tol=1e-6)

  def test_multihot_dp_input(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   combiner='sum',
                                   use_custom_kernels=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, combiner='sum', strategy='basic')

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes, hotness=5)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs, fwd_tol=1e-6)

  def test_multihot_dp_input_split(self):
    # More workers than tables
    table_sizes = gen_table_sizes(num_tables=max(hvd.size() // 2, 1))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   combiner='sum',
                                   use_custom_kernels=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, combiner='sum', strategy='basic')

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes, hotness=5)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs, fwd_tol=1e-6)

  def test_multihot_offloaded_mp_input(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   combiner='sum',
                                   use_custom_kernels=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    combiner='sum',
                                    strategy='basic',
                                    dp_input=False,
                                    gpu_embedding_size=0)

    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[hvd.rank()]

    dp_inputs, mp_inputs = gen_inputs(self.global_batch,
                                      table_sizes,
                                      mp_input_ids=mp_input_ids,
                                      hotness=5)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs, fwd_tol=1e-6)

  def test_multihot_offloaded_dp_input(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   combiner='sum',
                                   use_custom_kernels=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    combiner='sum',
                                    strategy='basic',
                                    gpu_embedding_size=0)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes, hotness=5)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs, fwd_tol=1e-6)


class DistributedEmbeddingModelFitTest(keras_parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    initialize_hvd()
    self.global_batch = 12

  def run_and_test(self):
    raise NotImplementedError

  def test_model_fit_bce(self):
    table_sizes = gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.5, momentum=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    label = tf.fill([self.global_batch // hvd.size(), 5], 0.5)

    dp_inputs, _ = gen_inputs(self.global_batch, table_sizes)
    ref_model(dp_inputs)

    # patched DistributedOptimizer that register local variables on first allreduce automatically
    dist_optimizer = dmp.DistributedOptimizer(optimizer)
    test_model.compile(optimizer=dist_optimizer, loss=bce)

    # need to force init so we can set_weight from reference and compare result
    # no need to broadcast since weight will be overwritten anyway
    test_model.fit(dp_inputs, label, epochs=1, steps_per_epoch=1)

    # broadcast ref model weights and set test model weights
    hvd.broadcast_variables(ref_model.variables, root_rank=0)
    ref_weights = ref_model.get_weights()
    num_tables = len(ref_model.embeddings)

    test_model.dist_embeddings.set_weights(ref_weights[:num_tables])
    test_model.dense.set_weights(ref_weights[num_tables:])

    with tf.GradientTape() as tape:
      ref_loss = bce(label, ref_model(dp_inputs))
    tape = hvd.DistributedGradientTape(tape)
    ref_grads = tape.gradient(ref_loss, ref_model.variables)
    optimizer.apply_gradients(zip(ref_grads, ref_model.variables))
    ref_weights = ref_model.get_weights()

    test_history = test_model.fit(dp_inputs, label, epochs=1, steps_per_epoch=1)
    test_weights = test_model.dist_embeddings.get_weights(True) + test_model.dense.get_weights()

    self.assertAllClose(ref_loss, test_history.history['loss'][0])
    for ref_w, test_w in zip(ref_weights, test_weights):
      # assert close here since order of accumulations(inputs and batch dim) might have changed
      self.assertAllClose(tf.convert_to_tensor(ref_w), tf.convert_to_tensor(test_w))

  def test_broadcast_callback(self):
    tf.keras.utils.set_random_seed(hvd.rank())
    num_tables = 7
    table_sizes = [[11, 7], [5, 8], [3, 8], [5, 8], [12, 25], [3, 12], [7, 13]]

    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')

    # create same input on all worker
    dup_ids = [
        tf.random.uniform(shape=[3], minval=0, maxval=table_sizes[i][0], dtype=tf.int64)
        for i in range(num_tables)
    ]
    dup_ids = hvd.broadcast_object(dup_ids, root_rank=0)

    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.5, momentum=0)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    label = tf.fill([3, 5], 0.3)
    # optimizer should not matter since broadcast happens after
    test_model.compile(optimizer=optimizer, loss=bce)
    callback = dmp.BroadcastGlobalVariablesCallback(0)

    # run one batch with broadcasting callback
    test_history = test_model.fit(dup_ids, label, epochs=1, steps_per_epoch=1, callbacks=[callback])

    # losses from initial batch should be different
    loss = test_history.history['loss'][0]
    losses = tf.unstack(hvd.allgather(tf.expand_dims(loss, axis=0)))
    for loss in losses[1:]:
      self.assertNotAllClose(losses[0], loss)

    # same output from each worker after broadcast data parallel weights
    test_out = test_model(dup_ids)
    test_outs = tf.unstack(hvd.allgather(tf.expand_dims(test_out, axis=0)))
    for out1 in test_outs[1:]:
      self.assertAllEqual(test_outs[0], out1)

    # now try model fit again, loss should be same
    test_history = test_model.fit(dup_ids, label, epochs=1, steps_per_epoch=1)
    loss = test_history.history['loss'][0]
    losses = tf.unstack(hvd.allgather(tf.expand_dims(loss, axis=0)))
    for loss in losses[1:]:
      # TODO(deyuf): understand why model.fit causes 1e-8 error sometime
      self.assertAllCloseAccordingToType(losses[0], loss)


def get_variable_length_ragged_test_data():
  return [
      tf.ragged.constant([[11, 12], [13, 14, 15], [16], [17], [21],
                          [22, 23, 24, 25, 26, 27, 28, 29, 210], [211], [212]]),
      tf.ragged.constant([[31, 32, 33], [34], [35], [36], [41, 42, 43, 44, 45, 46, 47, 48], [49],
                          [410], [411]]),
      tf.ragged.constant([[51], [52, 53, 54, 55, 56, 57, 58, 59, 510], [511], [512],
                          [61, 62, 63, 64, 65, 66, 67], [68], [69], [610]])
  ]


class DpToMpInputTest(keras_parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    initialize_hvd()

    tf.config.run_functions_eagerly(not flags.FLAGS.graph_mode)

    if flags.FLAGS.mixed_precision_policy:
      policy = tf.keras.mixed_precision.Policy(flags.FLAGS.mixed_precision_policy)
      tf.keras.mixed_precision.set_global_policy(policy)

  def run_and_test(self, features, rank_to_local_features=None):
    features = dict(enumerate(features))

    if rank_to_local_features is None:
      # generate a round-robin strategy
      rank_to_local_features = defaultdict(list)
      for feature_id in features.keys():
        rank = feature_id % hvd.size()
        rank_to_local_features[rank].append(feature_id)

    dp_inputs = {}
    for feature_id, feature in features.items():
      local_batch = feature.shape[0] // hvd.size()
      begin = hvd.rank() * local_batch
      end = begin + local_batch
      dp_inputs[feature_id] = feature[begin:end, ...]

    features_mp = _dp_to_mp_input(dp_inputs=dp_inputs,
                                  rank_to_local_features=rank_to_local_features)

    for feature_id in rank_to_local_features[hvd.rank()]:
      feature_mp = features_mp[feature_id]
      orig_data = features[feature_id]
      self.assertAllEqual(feature_mp, orig_data)

  def test_dense_onehot_dp_to_mp(self):
    table_sizes = gen_table_sizes(num_tables=41)
    dense_inputs = gen_inputs(global_batch=8, table_sizes=table_sizes, return_global=True)
    self.run_and_test(dense_inputs)

  def test_dense_multihot_dp_to_mp(self):
    table_sizes = gen_table_sizes(num_tables=41)
    dense_inputs = gen_inputs(global_batch=8,
                              table_sizes=table_sizes,
                              hotness=5,
                              return_global=True)
    self.run_and_test(dense_inputs)

  def test_ragged_dp_to_mp(self):
    self.run_and_test(get_variable_length_ragged_test_data())

  def test_ragged_dp_to_mp_unbalanced(self):
    ragged_data = get_variable_length_ragged_test_data()
    rank_to_local_features = {}
    for rank in range(hvd.size()):
      if rank == 0:
        # corner-case test - send all features to a single worker
        rank_to_local_features[rank] = list(range(len(ragged_data)))
      else:
        rank_to_local_features[rank] = []

    self.run_and_test(ragged_data, rank_to_local_features)

  def test_ragged_and_dense_dp_to_mp(self):
    dense_inputs = gen_inputs_onehot(global_batch=8,
                                     table_sizes=[[10, 8], [100, 8], [1000, 8], [10, 16], [10, 16],
                                                  [10, 4]],
                                     return_global=True)
    all_inputs = [*dense_inputs, *get_variable_length_ragged_test_data()]
    self.run_and_test(all_inputs)

  def test_ragged_and_dense_dp_to_mp_reversed(self):
    dense_inputs = gen_inputs_onehot(global_batch=8,
                                     table_sizes=[[10, 8], [100, 8], [1000, 8], [10, 16], [10, 16],
                                                  [10, 4]],
                                     return_global=True)
    all_inputs = [*get_variable_length_ragged_test_data(), *dense_inputs]
    self.run_and_test(all_inputs)


if __name__ == "__main__":
  test.main()
