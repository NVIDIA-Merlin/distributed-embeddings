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
import time
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.keras import keras_parameterized
import horovod.tensorflow as hvd
from distributed_embeddings.python.layers import dist_model_parallel as dmp

# There are some functions in TF that pylint can't inspect correctly which leads to incorrect
# report of unexpected-keyword-arg, no-value-for-parameter. Disable them globally here
# pylint: disable=no-self-use,unexpected-keyword-arg,no-value-for-parameter,missing-docstring


class EmbeddingListModel(tf.keras.Model):
  """A simple model for test"""

  def __init__(self,
               table_sizes,
               distribute=False,
               strategy='basic',
               dp_input=True,
               input_table_map=None,
               column_slice_threshold=None):
    super().__init__()
    self.embeddings = []
    for size in table_sizes:
      self.embeddings.append(tf.keras.layers.Embedding(*size))
    if distribute:
      self.dist_embeddings = dmp.DistributedEmbedding(self.embeddings,
                                                      strategy=strategy,
                                                      dp_input=dp_input,
                                                      input_table_map=input_table_map,
                                                      column_slice_threshold=column_slice_threshold)
    else:
      self.dist_embeddings = None
    self.dense = tf.keras.layers.Dense(5)
    self.input_table_map = input_table_map

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

  def train_step(self, data):
    with tf.GradientTape() as tape:
      out = tf.reduce_sum(self(data[0]))
    tape = dmp.DistributedGradientTape(tape)
    gradients = tape.gradient(out, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    return {"out": out}


class DistributedEmbeddingTest(keras_parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
      tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    self.hvd_rank = hvd.rank()
    self.hvd_size = hvd.size()
    seed = int(time.time())
    self.seed = hvd.broadcast_object(seed, root_rank=0)
    self.global_batch = 24

  def gen_table_sizes(self, num_tables=None):
    random.seed(self.seed)
    if num_tables is None:
      num_tables = random.randint(self.hvd_size, 2 * self.hvd_size)
    table_sizes = []
    for _ in range(num_tables):
      table_height = random.randint(3, 20)
      table_width = random.randint(3, 15)
      table_sizes.append([table_height, table_width])
    return table_sizes

  def gen_mapping(self, num_tables):
    # make sure each table have input
    mapping = list(range(num_tables))
    # fix number of shared input to 3
    for _ in range(3):
      mapping.append(random.randint(0, num_tables - 1))
    random.shuffle(mapping)
    return mapping

  def gen_inputs(self, table_sizes, input_to_table_map=None, mp_input_ids=None):
    # create global inputs
    if input_to_table_map is None:
      input_to_table_map = list(range(len(table_sizes)))
    global_inputs = [
        tf.random.uniform(shape=[self.global_batch],
                          minval=0,
                          maxval=table_sizes[i][0],
                          dtype=tf.int64) for i in input_to_table_map
    ]
    for t in global_inputs:
      hvd.broadcast(t, root_rank=0)
    local_batch = self.global_batch // self.hvd_size
    dp_inputs = [
        t[self.hvd_rank * local_batch:(self.hvd_rank + 1) * local_batch] for t in global_inputs
    ]
    mp_inputs = [global_inputs[i] for i in mp_input_ids] if mp_input_ids else None

    return dp_inputs, mp_inputs

  def run_and_test(self, ref_model, ref_inputs, test_model, test_inputs):
    tf.keras.utils.set_random_seed(int(time.time()) + self.hvd_rank)
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
    self.assertAllEqual(ref_out, test_out)

    # slicing grad is tricky. so we check weights updated with grad
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.5, momentum=0)
    optimizer.apply_gradients(zip(ref_grads, ref_model.variables))
    optimizer.apply_gradients(zip(test_grads, test_model.variables))
    ref_weights = ref_model.get_weights()
    test_weights = test_model.dist_embeddings.get_weights(True) + test_model.dense.get_weights()

    for ref_w, test_w in zip(ref_weights, test_weights):
      # assert close here since order of accumulations(inputs and batch dim) might have changed
      self.assertAllClose(tf.convert_to_tensor(ref_w), tf.convert_to_tensor(test_w))

  def test_broadcast(self):
    tf.keras.utils.set_random_seed(int(time.time()) + self.hvd_rank)
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
    table_sizes = self.gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')

    dp_inputs, _ = self.gen_inputs(table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_memory_optimized(self):
    table_sizes = self.gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='memory_optimized')

    dp_inputs, _ = self.gen_inputs(table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_shared_basic(self):
    table_sizes = self.gen_table_sizes()
    input_to_table_map = self.gen_mapping(len(table_sizes))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   input_table_map=input_to_table_map)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    input_table_map=input_to_table_map)

    dp_inputs, _ = self.gen_inputs(table_sizes, input_to_table_map)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_shared_basic_mp(self):
    table_sizes = self.gen_table_sizes()
    input_to_table_map = self.gen_mapping(len(table_sizes))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   input_table_map=input_to_table_map)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    dp_input=False,
                                    input_table_map=input_to_table_map)

    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[self.hvd_rank]
    dp_inputs, mp_inputs = self.gen_inputs(table_sizes, input_to_table_map, mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_shared_mb_mp(self):
    table_sizes = self.gen_table_sizes()
    input_to_table_map = self.gen_mapping(len(table_sizes))

    ref_model = EmbeddingListModel(table_sizes,
                                   distribute=False,
                                   input_table_map=input_to_table_map)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    dp_input=False,
                                    input_table_map=input_to_table_map)

    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[self.hvd_rank]
    dp_inputs, mp_inputs = self.gen_inputs(table_sizes, input_to_table_map, mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_column_slice_merge(self):
    # test on 4 GPUs
    table_sizes = [[100, 8], [5, 8], [10, 8], [25, 4]]
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    column_slice_threshold=45)

    dp_inputs, _ = self.gen_inputs(table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)
    for tables in test_model.dist_embeddings.strategy.table_ids_list:
      self.assertEqual(len(tables), len(set(tables)))

  def test_column_slice_threshold(self):
    table_sizes = self.gen_table_sizes()
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='basic',
                                    column_slice_threshold=30)

    dp_inputs, _ = self.gen_inputs(table_sizes)
    self.run_and_test(ref_model, dp_inputs, test_model, dp_inputs)

  def test_column_slice_dup_worker(self):
    table_sizes = [[10, 4], [11, 2], [4, 2], [4, 2]]
    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes,
                                    distribute=True,
                                    strategy='memory_balanced',
                                    dp_input=False,
                                    column_slice_threshold=10)
    mp_input_ids = test_model.dist_embeddings.strategy.input_ids_list[self.hvd_rank]
    dp_inputs, mp_inputs = self.gen_inputs(table_sizes, mp_input_ids=mp_input_ids)
    self.run_and_test(ref_model, dp_inputs, test_model, mp_inputs)

  def test_model_fit_basic(self):
    table_sizes = self.gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')
    optimizer = tf.keras.optimizers.SGD(learning_rate=1.5, momentum=0)
    test_model.compile(optimizer=optimizer)

    dp_inputs, _ = self.gen_inputs(table_sizes)
    ref_model(dp_inputs)
    test_model(dp_inputs)
    # broadcast ref model weights and set test model weights
    hvd.broadcast_variables(ref_model.variables, root_rank=0)
    ref_weights = ref_model.get_weights()
    num_tables = len(ref_model.embeddings)
    test_model.dist_embeddings.set_weights(ref_weights[:num_tables])
    test_model.dense.set_weights(ref_weights[num_tables:])

    with tf.GradientTape() as tape:
      ref_out = tf.reduce_sum(ref_model(dp_inputs))
    tape = hvd.DistributedGradientTape(tape)
    ref_grads = tape.gradient(ref_out, ref_model.variables)

    optimizer.apply_gradients(zip(ref_grads, ref_model.variables))
    ref_weights = ref_model.get_weights()

    test_history = test_model.fit(dp_inputs, epochs=1, steps_per_epoch=1)
    test_weights = test_model.dist_embeddings.get_weights(True) + test_model.dense.get_weights()

    self.assertAllClose(ref_out, test_history.history['out'][0])
    for ref_w, test_w in zip(ref_weights, test_weights):
      # assert close here since order of accumulations(inputs and batch dim) might have changed
      self.assertAllClose(tf.convert_to_tensor(ref_w), tf.convert_to_tensor(test_w))

  def test_set_weight_uninitialized(self):
    table_sizes = self.gen_table_sizes()

    ref_model = EmbeddingListModel(table_sizes, distribute=False)
    test_model = EmbeddingListModel(table_sizes, distribute=True, strategy='basic')

    dp_inputs, _ = self.gen_inputs(table_sizes)

    # run a batch to initialize weight tensors
    _ = ref_model(dp_inputs)
    ref_weights = ref_model.get_weights()
    num_tables = len(ref_model.embeddings)
    with self.assertRaises(ValueError):
      test_model.dist_embeddings.set_weights(ref_weights[:num_tables])
      test_model.dense.set_weights(ref_weights[num_tables:])


if __name__ == "__main__":
  test.main()
