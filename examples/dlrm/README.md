# Benchmarks of MLPerf DLRM

[Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) running on MLPerf [configuration](https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow2/Recommendation/DLRM)

| GPUS   | Precision | Global Batch | Samples/sec | Epoch Time | AUC     |
| ------ | --------- | ------------ | ----------- | ---------- | ------- |
| 8xA100 | TF32      | 65536        | 9157869     | 7m38s      | 0.80248 |
| 8xA100 | AMP       | 65536        | 10416232    | 6m43s      | 0.80262 |

###
