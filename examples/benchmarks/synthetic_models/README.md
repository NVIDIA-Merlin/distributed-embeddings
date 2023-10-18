# Synthetic Model Example

Because there is no publicly available dataset that can support creating large state of art recommenders (The largest public one is [Criteo 1 TB](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/)). We create "synthetic" models based on synthetic data as example of how to create large recommender models by distributed-embeddings, also serve as performance benchmarks.

## Models

There are 6 different size of models defined in [synthetic_models.py](synthetic_models.py):

| Model    | Total number of embedding tables | Total embedding size (GiB) |
| -------- | :------------------------------: | :------------------------: |
| Tiny     |                55                |            4.2             |
| Small    |               107                |            26.3            |
| Medium   |               311                |           206.2            |
| Large    |               612                |           773.8            |
| Jumbo    |               1022               |           3109.5           |
| Colossal |               2002               |          22327.4           |

Detailed config can be found in [config_v3.py](config_v3.py).

## Run benchmarks

**Single GPU:**

```python
python main.py --model small --optimizer sgd --batch_size 65536
```

**Multiple GPU:**

```python
horovodrun -np 32 python main.py --model large --optimizer adagrad --batch_size 65536 --column_slice_threshold $((1280*1048576))
```

**Arguments:**

```shell
main.py:
  --alpha: Exponent to generate power law distributed data
    (default: '1.05')
    (a number)
  --batch_size: Global batch size
    (default: '4')
    (an integer)
  --column_slice_threshold: Upper bound of elements count in each column slice
    (an integer)
  --[no]dp_input: Use data parallel input
    (default: 'false')
  --model: Choose model size to run benchmark
    (default: 'tiny')
  --num_data_batches: Number of batches of synthetic data to generate
    (default: '10')
    (an integer)
  --num_steps: Number of steps to benchmark
    (default: '100')
    (an integer)
  --optimizer: <sgd|adagrad|adam>: Optimizer
    (default: 'sgd')
```

## Preliminary Benchmark Results

**Setup:**

* System: DGX-A100
* Run config:
  * Global batch bize: 65536
  * Optimizer: Adagrad

| time (ms) |  1GPU  |  8GPU  | 16GPU  | 32GPU  | 128GPU |
| --------- | :----: | :----: | :----: | :----: | :----: |
| Tiny      | 24.433 | 5.537  | 4.867  |        |        |
| Small     | 67.355 | 17.203 | 12.461 | 11.839 |        |
| Medium    |        | 63.393 | 46.636 | 37.732 | 27.329 |
| Large     |        |        |        | 67.57  | 37.934 |
| Jumbo     |        |        |        |        | 124.3  |
