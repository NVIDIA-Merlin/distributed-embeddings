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
"""DLRM Sample using model parallel embedding"""

import sys
import os
import json
import math
import numpy as np
from absl import app, flags
import tensorflow as tf
from tensorflow.keras import initializers
import horovod.tensorflow as hvd
from utils import dot_interact, RawBinaryDataset, LearningRateScheduler, DummyDataset, DLRMInitializer
from distributed_embeddings.python.layers import dist_model_parallel as dmp
from distributed_embeddings.python.layers import embedding
# pylint: disable=abstract-method, unused-argument


def define_command_line_flags():
  flags.DEFINE_string("dataset_path",
                      default=None,
                      help="Path to the JSON file with the sizes of embedding tables")
  flags.DEFINE_float("learning_rate", default=24, help="Learning rate")
  flags.DEFINE_integer("batch_size", default=64 * 1024, help="Global batch size used for training")
  flags.DEFINE_list("top_mlp_dims", [1024, 1024, 512, 256, 1], "Linear layer sizes for the top MLP")
  flags.DEFINE_list("bottom_mlp_dims", [512, 256, 128], "Linear layer sizes for the bottom MLP")
  flags.DEFINE_integer("num_numerical_features",
                       default=13,
                       help='Number of numerical features to be read from the dataset. '
                       'If set to 0, then no numerical features will be loaded '
                       'and the Bottom MLP will not be evaluated')
  flags.DEFINE_integer("num_categorical_features",
                       default=26,
                       help='Same as number of one-hot tables for now.')
  flags.DEFINE_integer('num_batches',
                       default=340,
                       help='Number of training batches in the synthetic dataset')
  flags.DEFINE_list('table_sizes',
                    default=26 * [1000],
                    help='Number of categories for each embedding table of the synthetic dataset')
  flags.DEFINE_integer("embedding_dim",
                       default=128,
                       help='Number of columns in the embedding tables')
  flags.DEFINE_bool("dp_input", default=False, help="Use data parallel input")
  flags.DEFINE_bool("test_combiner", default=False, help="Use embedding implementation for testing")
  flags.DEFINE_string("dist_strategy", default='memory_balanced', help="distribution strategy")


os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'
define_command_line_flags()
FLAGS = flags.FLAGS
app.define_help_flags()
app.parse_flags_with_usage(sys.argv)

if FLAGS.dataset_path is not None:
  with open(os.path.join(FLAGS.dataset_path, 'model_size.json'), 'r', encoding='utf-8') as f:
    global_table_sizes = json.load(f)
    global_table_sizes = list(global_table_sizes.values())
    global_table_sizes = [s + 1 for s in global_table_sizes]
    FLAGS.table_sizes = global_table_sizes


class DLRM(tf.keras.Model):
  """DLRM model
  """

  def __init__(self, model_flags):
    super().__init__()
    self.table_sizes = model_flags.table_sizes
    self.embedding_dim = model_flags.embedding_dim
    self.bottom_mlp_dims = [int(d) for d in model_flags.bottom_mlp_dims]
    self.top_mlp_dims = [int(d) for d in model_flags.top_mlp_dims]
    self.distributed = hvd.size() > 1 if hvd.is_initialized() else False
    self._create_bottom_mlp()
    self._create_top_mlp()
    self._create_embeddings(model_flags)

  def call(self, inputs):
    numerical_features, cat_features = inputs
    for l in self.bottom_mlp_layers:
      numerical_features = l(numerical_features)
    if self.distributed:
      embedding_outputs = self.dist_embedding(cat_features)
    else:
      embedding_outputs = [e(i) for e, i in zip(self.embedding_layers, cat_features)]
    x = dot_interact(embedding_outputs, numerical_features)
    for l in self.top_mlp_layers:
      x = l(x)
    return x

  def _create_embeddings(self, model_flags):
    self.embedding_layers = []
    for table_size in self.table_sizes:
      if model_flags.test_combiner:
        self.embedding_layers.append(
            embedding.Embedding(input_dim=table_size,
                                output_dim=self.embedding_dim,
                                embeddings_initializer=DLRMInitializer(),
                                combiner='sum'))
      else:
        self.embedding_layers.append(
            tf.keras.layers.Embedding(input_dim=table_size,
                                      output_dim=self.embedding_dim,
                                      embeddings_initializer=DLRMInitializer()))
    if self.distributed:
      self.dist_embedding = dmp.DistributedEmbedding(self.embedding_layers,
                                                     strategy=model_flags.dist_strategy,
                                                     dp_input=model_flags.dp_input)

  def _create_bottom_mlp(self):
    self.bottom_mlp_layers = []
    for dim in self.bottom_mlp_dims:
      self.bottom_mlp_layers.append(
          tf.keras.layers.Dense(dim,
                                activation='relu',
                                kernel_initializer=initializers.GlorotNormal(),
                                bias_initializer=initializers.RandomNormal(stddev=math.sqrt(1. /
                                                                                            dim))))

  def _create_top_mlp(self):
    self.top_mlp_layers = []
    for dim in self.top_mlp_dims[:-1]:
      self.top_mlp_layers.append(
          tf.keras.layers.Dense(dim,
                                activation='relu',
                                kernel_initializer=initializers.GlorotNormal(),
                                bias_initializer=initializers.RandomNormal(stddev=math.sqrt(1. /
                                                                                            dim))))
    self.top_mlp_layers.append(
        tf.keras.layers.Dense(self.top_mlp_dims[-1],
                              activation='linear',
                              kernel_initializer=initializers.GlorotNormal(),
                              bias_initializer=initializers.RandomNormal(
                                  stddev=math.sqrt(1. / self.top_mlp_dims[-1]))))


def main(argv):

  hvd.init()
  gpus = tf.config.experimental.list_physical_devices('GPU')
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

  dlrm_model = DLRM(FLAGS)

  if FLAGS.dp_input:
    table_ids = list(range(len(FLAGS.table_sizes)))
  else:
    table_ids = dlrm_model.dist_embedding.strategy.input_ids_list[hvd.rank()]

  if FLAGS.dataset_path is not None:
    train_dataset = RawBinaryDataset(data_path=FLAGS.dataset_path,
                                     batch_size=FLAGS.batch_size,
                                     numerical_features=FLAGS.num_numerical_features,
                                     categorical_features=table_ids,
                                     categorical_feature_sizes=FLAGS.table_sizes,
                                     prefetch_depth=10,
                                     drop_last_batch=True,
                                     offset=FLAGS.batch_size // hvd.size() * hvd.rank(),
                                     lbs=FLAGS.batch_size // hvd.size(),
                                     dp_input=FLAGS.dp_input)
    eval_dataset = RawBinaryDataset(data_path=FLAGS.dataset_path,
                                    valid=True,
                                    batch_size=FLAGS.batch_size,
                                    numerical_features=FLAGS.num_numerical_features,
                                    categorical_features=table_ids,
                                    categorical_feature_sizes=FLAGS.table_sizes,
                                    prefetch_depth=10,
                                    drop_last_batch=True,
                                    offset=FLAGS.batch_size // hvd.size() * hvd.rank(),
                                    lbs=FLAGS.batch_size // hvd.size(),
                                    dp_input=FLAGS.dp_input)
  else:
    train_dataset = DummyDataset(FLAGS, hvd.size(), len(table_ids), True, FLAGS.dp_input)
    eval_dataset = DummyDataset(FLAGS, hvd.size(), len(table_ids), False, FLAGS.dp_input)

  optimizer = tf.keras.optimizers.SGD(lr=FLAGS.learning_rate, momentum=0)
  scheduler = LearningRateScheduler([optimizer],
                                    warmup_steps=8000,
                                    base_lr=FLAGS.learning_rate,
                                    decay_start_step=48000,
                                    decay_steps=24000)
  bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                           from_logits=True)

  @tf.function
  def train_step(numerical_features, categorical_features, labels):
    scheduler()
    with tf.GradientTape() as tape:
      predictions = dlrm_model((numerical_features, categorical_features))
      loss = tf.math.reduce_mean(bce(labels, predictions))
    tape = dmp.DistributedGradientTape(tape)
    gradients = tape.gradient(loss, dlrm_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, dlrm_model.trainable_variables))
    return loss

  for step, (numerical_features, categorical_features, labels) in enumerate(train_dataset):
    if FLAGS.test_combiner:
      categorical_features = [tf.reshape(c_f, [-1, 1]) for c_f in categorical_features]
    loss = train_step(numerical_features, categorical_features, labels)
    if step == 0:
      dmp.broadcast_variables(dlrm_model.variables, root_rank=0)
    if step % 1000 == 0:
      loss = hvd.allreduce(loss, name="mean_loss", op=hvd.Average)
      print("step: ", step, " loss: ", loss)

  # eval
  auc_metric = tf.keras.metrics.AUC(num_thresholds=8000,
                                    curve='ROC',
                                    summation_method='interpolation',
                                    name='my_auc')
  bce_op = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE,
                                              from_logits=False)
  all_test_losses = []
  for step, (numerical_features, categorical_features, labels) in enumerate(eval_dataset):
    if FLAGS.test_combiner:
      categorical_features = [tf.reshape(c_f, [-1, 1]) for c_f in categorical_features]
    predictions = tf.math.sigmoid(dlrm_model((numerical_features, categorical_features)))
    predictions = hvd.allgather(predictions)

    if hvd.rank() == 0:
      auc_metric.update_state(labels, predictions)
      all_test_losses.append(bce_op(labels, predictions))

  if hvd.rank() == 0:
    auc = auc_metric.result().numpy().item()
    test_loss = tf.reduce_mean(all_test_losses).numpy().item()
    print(f'Evaluation completed, AUC: {auc}, test_loss: {test_loss}')

  # save out embedding weight
  full_embedding_weights = dlrm_model.dist_embedding.get_weights()
  if hvd.rank() == 0:
    np.savez(os.path.join('/tmp', 'embedding_weights'), *full_embedding_weights)


if __name__ == '__main__':
  app.run(main)
