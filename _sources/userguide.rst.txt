#################################################
Distributed Model Parallel
#################################################

*************************************************
Automatic model parallel wrapper
*************************************************

``distributed_embeddings.dist_model_parallel`` enables model parallel
training by changing only three lines of your script. It can also be
used alongside data parallel to form hybrid parallel training.
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

``dist_model_parallel`` can be applied on distributed-embeddings embedding layers,
keras embedding layers and any user defined custom embedding layer.

*************************************************
Large table exceeding single GPU's memory
*************************************************

If single embedding table exceeds GPU memory, or portion of GPU memory considering
optimizer's memory requirement, we have to split the embedding table and distribute
them to multiple GPU.
Currently distributed-embeddings supports column slicing embedding tables by passing
``column_slice_threshold`` to DistributedEmbedding, Example:

.. code:: python

   # Split embedding tables that are larger than 200000 elements (not Bytes)
   embedding_layers = dmp.DistributedEmbedding(embedding_layers, column_slice_threshold=200000)

Embedding will be evenly split into the smallest power of 2 number of slices so that each
slice is smaller than ``column_slice_threshold``.

Alternatively, user can specify ``row_slice_threshold``, example:

.. code:: python

   # Split embedding tables that are larger than 200000 elements (not Bytes)
   embedding_layers = dmp.DistributedEmbedding(embedding_layers, row_slice_threshold=200000)

Different from ``column_slice_threshold``, table larger than the threshold will be sliced
on the first dimension into 'rows'. Another difference is that table will be sliced evenly
into the num_worker slices and get distributed among all workers.
This is useful when user have super tall and narrow table.

*************************************************
Mix Matching Distribution Strategies
*************************************************

DistributedEmbedding have flexible api options allowing user specify how they want embedding layers to be distributed among GPUs. When multiple options are used, strategies will be applied in following order, balancing ease of use and fine grained control:

``data_parallel_threshold`` - Table smaller than the threshold will run in data parallel. This greatly reduces communication in case of small tables with large amount of lookups.

``row_slice_threshold`` - Table with more elements than it will be sliced into rows and distributed evenly onto all workers.

``column_slice_threshold`` - This is the most flexible option. Tables that aren't running in dp or row slice will get here and get sliced into columns smaller than column_slice_threshold.

We currently don't support partial participation on data parallel and row slice. So tables under those strategies will be distributed onto all workers. For the rest of tables, some may have been column sliced, one of the following strategies will apply to distribute them with model parallel:

``basic`` - round-robin distribute table slices in original order

``memory_balanced`` - round-robin distribute table slices by size order. This mode balances compute and memory.

``memory_optimized`` - distribute table slices to achieve most even memory usage. This mode helps avoid OOM in workloads with skewed tables sizes.

**In summary:**

1. Small tables run data parallel on all workers
2. Largest tables get evenly row slied onto all workers
3. All other tables run in model parallel, potentially after 2-way to max workers way column slice


*************************************************
Shared Embedding
*************************************************

It is common that some features share embedding. For example, watched video and browsed video can
share video embedding. User can supports this case by passing ``input_table_map`` at intialization
time, example:

.. code:: python

   # The first and the last input both map to embedding 0
   embedding_layers = dmp.DistributedEmbedding(
       embedding_layers,
       input_table_map=[0, 1, 2, 3, 0])


#################################################
Embedding Layers
#################################################

``distributed_embeddings.Embedding`` combines functionalities of
``tf.keras.layers.Embedding`` and ``tf.nn.embedding_lookup_sparse``
under a unified Keras layer API. The backend is designed to achieve high
GPU efficiency. Two kinds of inputs are supported. We call them
fixed/variable hotness as opposite to confusing dense/sparse term
various TF API uses. The difference is whether all sample in the batch
contains same number of indices.
Fixed hotness inputs are regular ``Tensor`` while variable hotness
inputs are 2D ``RaggedTensor`` or ``SparseTensor``. Elements of inputs are
ids to be looked up. Lookup output from inner most dimension are considered
from same sample and will be reduced if combiner is used.

Examples:

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

   >>> vhot_input = tf.ragged.constant([[1, 3, 1], [87], [5, 9], [6], [929]], dtype=tf.int64)
   >>> print(vhot_input.shape)
   (5, None)
   >>> layer = Embedding(1000, 64)
   >>> print(layer(vhot_input).shape)
   (5, None, 64)
   >>> layer = Embedding(1000, 64, combiner='sum')
   >>> print(layer(vhot_input).shape)
   (5, 64)

#################################################
Input Hashing
#################################################

A preprocessing layer that maps integer features to contiguous ranges. This layer extends ``tf.keras.layers.IntegerLookup`` with following functionalities:

1. Generates vocabulary on the fly so that training can start with empty vocabulary
2. Suport both CPU and GPU with efficient backends
3. Frequency of input keys are counted when GPU backend is used
4. Overflow protection. When lookup table grows beyond user-defined limit, new keys will be treat as OOV tokens and get mapped to 0.

With this, user can start or continugous train on new data, without offline data preprocessing.

.. code:: python

   lookup_layer = IntegerLookup(max_vocab_size)
   embedding_layer = tf.keras.layers.Embedding(max_vocab_size, embedding_width)
   ...
   # inside call() function
   input_ids = lookup_layer(input_hash_keys)
   embeddings = embedding_layer(input_ids)

For more details, see our Criteo `Example <https://github.com/NVIDIA-Merlin/distributed-embeddings/blob/main/examples/criteo/main.py>`_
and read TensorFlow `Preprocessing Layer Document <https://www.tensorflow.org/guide/migrate/migrating_feature_columns>`_
