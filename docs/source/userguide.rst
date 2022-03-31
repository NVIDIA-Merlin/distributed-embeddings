Distributed Model Parallel
==========================

``distributed_embeddings.dist_model_parallel`` is a tool to enable model parallel
training by changing only three lines of your script. It can also be
used alongside data parallel to form hybrid parallel training. Users can
easily experiment large scale embeddings beyond single GPU’s memory
capacity without complex code to handle cross-worker communication.
Example:

.. code:: python

   import dist_model_parallel as dmp

   class MyEmbeddingModel(tf.keras.Model):
     def  __init__(self):
       ...
       self.embedding_layers = [tf.keras.layers.Embedding(*size) for size in table_sizes]
       # add this line to wrap list of embedding layers used in the model
       self.embedding_layers = dmp.DistributedEmbedding(self.embedding_layers)
     def call(self, inputs):
       # embedding_outputs = [e(i) for e, i in zip(self.embedding_layers, inputs)]
       embedding_outputs = self.embedding_layers(inputs)
       ...

To work with Horovod data parallel, replace Horovod ``GradientTape`` and
broadcast. Take following example directly from Horovod
`documents <https://horovod.readthedocs.io/en/stable/tensorflow.html>`__:

.. code:: python

   @tf.function
   def training_step(inputs, labels, first_batch):
     with tf.GradientTape() as tape:
       probs = model(inputs)
       loss_value = loss(labels, probs)

     # Change Horovod Gradient Tape to dmp tape
     # tape = hvd.DistributedGradientTape(tape)
     tape = dmp.DistributedGradientTape(tape)
     grads = tape.gradient(loss_value, model.trainable_variables)
     opt.apply_gradients(zip(grads, model.trainable_variables))

     if first_batch:
       # Change Horovod broadcast_variables to dmp's
       # hvd.broadcast_variables(model.variables, root_rank=0)
       dmp.broadcast_variables(model.variables, root_rank=0)
     return loss_value

``distributed_embeddings.dist_model_parallel`` can be applied both distributed-embeddings and
Tensorflow embedding layers.

Embedding Layers
================

``distributed_embeddings.Embedding`` combines functionalities of
``tf.keras.layers.Embedding`` and ``tf.nn.embedding_lookup_sparse``
under a unified Keras layer API. The backend is designed to achieve high
GPU efficiency. Two kinds of inputs are supported. We call them
fixed/variable hotness as opposite to confusing dense/sparse term
various TF API uses. The difference is whether all sample in the batch
contains same number of indices. Inputs are “`potentially ragged
tensor <https://www.tensorflow.org/api_docs/python/tf/RaggedTensor#potentially_ragged_tensors_2>`__”.
Fixed hotness inputs are regular ``Tensor`` while variable hotness
inputs are 2D ``RaggedTensor`` with inner ragged dimension. Elements of
inputs are ids to be looked up. Lookup output from inner most dimension
are considered from same sample and will be reduced if combiner is used.

Examples:
~~~~~~~~~

**One-hot embedding:**

.. code:: python

   >>> layer = Embedding(1000, 64)
   >>> onehot_input = tf.random.uniform(shape=(16, 1), maxval=1000, dtype=tf.int32)
   >>> print(layer(onehot_input).shape)
   (16, 1, 64)

**Fixed hotness embedding:**

.. code:: python

   >>> fixedhot_input = tf.random.uniform(shape=(16, 7), maxval=1000, dtype=tf.int32)
   >>> print(fixedhot_input.shape)
   (16, 7)
   >>> layer = Embedding(1000, 64)
   >>> print(layer(fixedhot_input).shape)
   (16, 7, 64)
   >>> layer = Embedding(1000, 64, combiner='mean')
   >>> print(layer(fixedhot_input).shape)
   (16, 64)

**Variable hotness embedding:**

.. code:: python

   >>> variablehot_input = tf.ragged.constant([[3, 1, 4, 1], [87], [5, 9, 2], [6], [929]], dtype=tf.int64)
   >>> print(variablehot_input.shape)
   (5, None)
   >>> layer = Embedding(1000, 64)
   >>> print(layer(variablehot_input).shape)
   (5, None, 64)
   >>> layer = Embedding(1000, 64, combiner='sum')
   >>> print(layer(variablehot_input).shape)
   (5, 64)

Larger than GPU memory table
============================

If single embedding table exceeds GPU memory, or portion of GPU memory
depends on the optimizer, we have to split the embedding table and
distribute them to multiple GPU. Currently distributed-embeddings supports column slicing
embedding tables by passing ``column_slice_threshold`` to
DistributedEmbedding, example:

.. code:: python

   # Split embedding tables that are larger than 20000000 elements (not Bytes)
   embedding_layers = dmp.DistributedEmbedding(embedding_layers, column_slice_threshold=20000000)

Embedding will be evenly split into the smallest power of 2 number of
slices so that each slice is smaller than ``column_slice_threshold``.

Shared Embedding
================

It is common that some features share embedding. For example, watched
video and browsed video can share video embedding. distributed-embeddings supports shared
embedding by passing a ``input_table_map`` to DistributedEmbedding,
example:

.. code:: python

   # The first and the last input both map to embedding 0
   embedding_layers = dmp.DistributedEmbedding(
       embedding_layers,
       input_table_map=[0, 1, 2, 3, 0])
