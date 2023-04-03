# [Distributed Embeddings](https://github.com/NVIDIA-Merlin/distributed-embeddings)

[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/distributed-embeddings/Introduction.html)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/NVTabular)](https://github.com/NVIDIA-Merlin/distributed-embeddingsb/blob/main/LICENSE)

distributed-embeddings is a library for building large embedding based (e.g. recommender) models in Tensorflow 2. It provides a scalable model parallel wrapper that automatically distribute embedding tables to multiple GPUs, as well as efficient embedding operations that cover and extend Tensorflow's embedding functionalities.

Refer to [NVIDIA Developer blog](https://developer.nvidia.com/blog/fast-terabyte-scale-recommender-training-made-easy-with-nvidia-merlin-distributed-embeddings/) about Terabyte-scale Recommender Training for more details.

## Features

### Distributed model parallel wrapper
`distributed_embeddings.dist_model_parallel` is a tool to enable model parallel training by changing only three lines of your script. It can also be used alongside data parallel to form hybrid parallel training. Users can easily experiment large scale embeddings beyond single GPU's memory capacity without complex code to handle cross-worker communication.

### Embedding Layers

`distributed_embeddings.Embedding` combines functionalities of `tf.keras.layers.Embedding` and `tf.nn.embedding_lookup_sparse` under a unified Keras layer API. The backend is designed to achieve high GPU efficiency.

See more details at [User Guide](https://nvidia-merlin.github.io/distributed-embeddings/userguide.html)

## Installation
### Requirements
Python 3, CUDA 11 or newer, TensorFlow 2
### Containers ###
You can build inside 22.03 or later NGC TF2 [image](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow):

Note: horovod v0.27 and TensorFlow 2.10, alternatively NGC 23.03 container, is required for building v0.3+
```bash
docker pull nvcr.io/nvidia/tensorflow:23.03-tf2-py3
```
### Build from source

After clone this repository, run:
```bash
git submodule update --init --recursive
make pip_pkg && pip install artifacts/*.whl
```
Test installation with:
```python
python -c "import distributed_embeddings"
```
You can also run [Synthetic](https://github.com/NVIDIA-Merlin/distributed-embeddings/tree/main/examples/benchmarks/synthetic_models) and [DLRM](https://github.com/NVIDIA-Merlin/distributed-embeddings/blob/main/examples/dlrm/main.py) examples.

## Feedback and Support

If you'd like to contribute to the library directly, see the [CONTRIBUTING.md](https://github.com/NVIDIA-Merlin/distributed-embeddings/blob/main/CONTRIBUTING.md). We're particularly interested in contributions or feature requests for our feature engineering and preprocessing operations. To further advance our Merlin Roadmap, we encourage you to share all the details regarding your recommender system pipeline in this [survey](https://developer.nvidia.com/merlin-devzone-survey).

If you're interested in learning more about how distributed-embeddings works, see [documentation]( https://nvidia-merlin.github.io/distributed-embeddings/Introduction.html).
