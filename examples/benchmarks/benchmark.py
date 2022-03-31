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
"""Simple example of embedding lookup performance benchmark"""

import time
import tensorflow as tf
import distributed_embeddings
#from distributed_embeddings import embedding_lookup
#from distributed_embeddings.python.layers import embedding, dist_model_parallel

voc, emb, batch, max_hotness = 1000000, 128, 16384, 500
ragged_iters = 100
sparse_iters = 10
# create COO and Ragged inputs for test
# there is no 0 in input data, in order to allow convertion to sparse
row_lengths = tf.random.uniform(shape=(batch,),
                                minval=1,
                                maxval=max_hotness + 1,
                                dtype=tf.dtypes.int64)
values = tf.random.uniform(shape=(tf.math.reduce_sum(row_lengths),),
                           minval=1,
                           maxval=voc,
                           dtype=tf.dtypes.int64)

ragged_inp = tf.RaggedTensor.from_row_lengths(values, row_lengths)
sparse_inp = tf.sparse.from_dense(ragged_inp.to_tensor())

weight = tf.Variable(tf.random.uniform([voc, emb], dtype=tf.dtypes.float32))
optimizer = tf.keras.optimizers.SGD()
#print(ragged_inp, sparse_inp)

#ragged_out = distributed_embeddings.embedding_lookup(weight, ragged_inp, combiner='mean')
#sparse_out = tf.nn.embedding_lookup_sparse(weight, sparse_inp, sp_weights=None, combiner='mean')
#print(tf.math.reduce_max(tf.math.abs((sparse_out-ragged_out))))
#print(ragged_out, sparse_out)

for _ in range(20):
  warmup = tf.random.uniform(shape=(1024, 10240))
  warmup = tf.reduce_sum(warmup)
print(warmup)

start = time.time()
for _ in range(ragged_iters):
  ragged_out = distributed_embeddings.embedding_lookup(weight, ragged_inp, combiner='mean')
print(ragged_out[0][0])
end = time.time()
print("ragged time:", (end - start) * 1000 / ragged_iters)

start = time.time()
with tf.GradientTape(persistent=True) as tape:
  ragged_out = distributed_embeddings.embedding_lookup(weight, ragged_inp, combiner='mean')
for _ in range(ragged_iters):
  dw = tape.gradient(ragged_out, weight)
print(ragged_out[0][0])
end = time.time()
print("ragged grad time:", (end - start) * 1000 / ragged_iters)

start = time.time()
for _ in range(ragged_iters):
  optimizer.apply_gradients([(dw, weight)])
print(weight[0][0])
end = time.time()
print("ragged SGD time:", (end - start) * 1000 / ragged_iters)

start = time.time()
for _ in range(sparse_iters):
  sparse_out = tf.nn.embedding_lookup_sparse(weight, sparse_inp, sp_weights=None, combiner='mean')
print(sparse_out[0][0])
end = time.time()
print("sparse time:", (end - start) * 1000 / sparse_iters)

start = time.time()
with tf.GradientTape(persistent=True) as tape:
  sparse_out = tf.nn.embedding_lookup_sparse(weight, sparse_inp, sp_weights=None, combiner='mean')
for _ in range(sparse_iters):
  dw = tape.gradient(sparse_out, weight)
print(sparse_out[0][0])
end = time.time()
print("sparse grad time:", (end - start) * 1000 / sparse_iters)

start = time.time()
for _ in range(sparse_iters):
  optimizer.apply_gradients([(dw, weight)])
print(weight[0][0])
end = time.time()
print("sparse SGD time:", (end - start) * 1000 / sparse_iters)
