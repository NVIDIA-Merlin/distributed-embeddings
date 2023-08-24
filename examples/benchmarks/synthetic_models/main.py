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
"""Benchmarks of synthetic models"""

import os
from time import time

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

import horovod.tensorflow.keras as hvd

from config_v3 import synthetic_models_v3
from synthetic_models import SyntheticModelTFDE, SyntheticModelNative, InputGenerator

from distributed_embeddings import dist_model_parallel as dmp

os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'

# pylint: disable=line-too-long
# yapf: disable
flags.DEFINE_integer("batch_size", 4, help="Global batch size")
flags.DEFINE_integer("num_data_batches", 1, help="Number of batches of synthetic data to generate")
flags.DEFINE_float("alpha", 1.05, help="Exponent to generate power law distributed data")
flags.DEFINE_integer("num_steps", 100, help="Number of steps to benchmark")
flags.DEFINE_bool("dp_input", False, help="Use data parallel input")
flags.DEFINE_string("model", "tiny", help="Choose model size to run benchmark")
flags.DEFINE_enum("optimizer", "sgd", ["sgd", "adagrad", "adam"], help="Optimizer")
flags.DEFINE_integer("column_slice_threshold", None, help="Upper bound of elements count in each column slice")
flags.DEFINE_bool("use_model_fit", False, help="Use Keras model.fit")
flags.DEFINE_string("embedding_device", "/GPU:0", help="device to place embedding. inputs are placed on same device")
flags.DEFINE_enum("embedding_api", "tfde", ["native", "tfde"], help="embedding to use.")
flags.DEFINE_bool("amp", False, help="Use mixed precision")
# yapf: enable
# pylint: enable=line-too-long

FLAGS = flags.FLAGS


def main(_):
  hvd.init()
  hvd_rank = hvd.rank()
  hvd_size = hvd.size()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  if FLAGS.amp:
    mixed_precision.set_global_policy('mixed_float16')

  if FLAGS.batch_size % hvd_size != 0:
    raise ValueError(F"Batch size ({FLAGS.batch_size}) is not divisible by world size ({hvd_size})")

  model_config = synthetic_models_v3[FLAGS.model]
  if FLAGS.embedding_api == "tfde":
    if FLAGS.embedding_device != "/GPU:0":
      raise ValueError(
          F"distributed-embeddings api is not supported on device {FLAGS.embedding_device}.")
    model = SyntheticModelTFDE(model_config,
                               column_slice_threshold=FLAGS.column_slice_threshold,
                               dp_input=FLAGS.dp_input)
  elif FLAGS.embedding_api == "native":
    if FLAGS.dp_input is False or FLAGS.column_slice_threshold is not None:
      raise ValueError(
          "Model parallel inputs and column slicing are not supported with native embedding api.")
    model = SyntheticModelNative(model_config, embedding_device=FLAGS.embedding_device)
  else:
    raise ValueError(F"Unknown embedding api {FLAGS.embedding_api}.")

  mp_input_ids = None if FLAGS.dp_input else model.embeddings.strategy.input_ids_list[hvd_rank]
  input_gen = InputGenerator(model_config,
                             FLAGS.batch_size,
                             alpha=FLAGS.alpha,
                             mp_input_ids=mp_input_ids,
                             num_batches=FLAGS.num_data_batches,
                             embedding_device=FLAGS.embedding_device)

  if FLAGS.optimizer == "sgd":
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.03, momentum=0)
  if FLAGS.optimizer == "adagrad":
    optimizer = tf.keras.optimizers.legacy.Adagrad()
  if FLAGS.optimizer == "adam":
    optimizer = tf.keras.optimizers.legacy.Adam()

  bce = keras.losses.BinaryCrossentropy(from_logits=True)
  if FLAGS.use_model_fit:
    optimizer = dmp.DistributedOptimizer(optimizer)
    model.compile(optimizer=optimizer, loss=bce)
    callbacks = [dmp.BroadcastGlobalVariablesCallback(0)]
    epochs = FLAGS.num_steps // FLAGS.num_data_batches
    model.fit(input_gen,
              epochs=epochs,
              batch_size=FLAGS.batch_size,
              callbacks=callbacks,
              verbose=1 if hvd_rank == 0 else 0)
    return

  # Not using model.fit() api. benchmark custom loop below
  # Run one step to init
  (numerical_features, cat_features), labels = input_gen[-1]
  model((numerical_features, cat_features))
  dmp.broadcast_variables(model.variables, root_rank=0)

  @tf.function
  def train_step(numerical_features, categorical_features, labels):
    with tf.GradientTape() as tape:
      predictions = model((numerical_features, categorical_features))
      loss = bce(labels, predictions)
    tape = dmp.DistributedGradientTape(tape)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

  # Run 5 steps to compile and warm up
  (numerical_features, cat_features), labels = input_gen[-1]
  for _ in range(5):
    loss = train_step(numerical_features, cat_features, labels)
  loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
  # printing initial loss here to force sync before we start timer
  print(F"Initial loss: {loss:.3f}")

  start = time()
  # Input data consumes a lot of memory. Instead of generating num_steps batch of synthetic data,
  # We generate smaller amount of data and loop over them
  for step in range(FLAGS.num_steps):
    inputs = input_gen[step % FLAGS.num_data_batches]
    (numerical_features, cat_features), labels = inputs
    loss = train_step(numerical_features, cat_features, labels)
    if step % 50 == 0:
      loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
      if hvd_rank == 0:
        print(F"Benchmark step [{step}/{FLAGS.num_steps}]")

  loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
  if hvd_rank == 0:
    # printing GPU tensor forces a sync. loss was allreduced, printing on one GPU is enough
    # for computing time so we don't print noisy messages from all ranks
    print(F"loss: {loss:.3f}")
    stop = time()
    print(F"Iteration time: {(stop - start) * 1000 / FLAGS.num_steps:.3f} ms")


if __name__ == '__main__':
  app.run(main)
