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
"""synthetic model configs"""

from collections import namedtuple

# nnz is a list of integer(s). If not shared total number of embedding tables will
# be num_tables * len(nnz)
EmbeddingConfig = namedtuple("EmbeddingConfig",
                             ["num_tables", "nnz", "num_rows", "width", "shared"])

# The the last MLP layer project to 1 should be omitted in mlp_sizes
# interact_stride is stride of 1d pooling which is used emulate memory limited interaction/FM
ModelConfig = namedtuple(
    "ModelConfig",
    ["name", "embedding_configs", "mlp_sizes", "num_numerical_features", "interact_stride"])

model_tiny = ModelConfig(name="Tiny V3",
                         embedding_configs=[
                             EmbeddingConfig(1, [1, 10], 10000, 8, True),
                             EmbeddingConfig(1, [1, 10], 1000000, 16, True),
                             EmbeddingConfig(1, [1, 10], 25000000, 16, True),
                             EmbeddingConfig(1, [1], 25000000, 16, False),
                             EmbeddingConfig(16, [1], 10, 8, False),
                             EmbeddingConfig(10, [1], 1000, 8, False),
                             EmbeddingConfig(4, [1], 10000, 8, False),
                             EmbeddingConfig(2, [1], 100000, 16, False),
                             EmbeddingConfig(19, [1], 1000000, 16, False),
                         ],
                         mlp_sizes=[256, 128],
                         num_numerical_features=10,
                         interact_stride=None)

model_small = ModelConfig(name="Small V3",
                          embedding_configs=[
                              EmbeddingConfig(5, [1, 30], 10000, 16, True),
                              EmbeddingConfig(3, [1, 30], 4000000, 32, True),
                              EmbeddingConfig(1, [1, 30], 50000000, 32, True),
                              EmbeddingConfig(1, [1], 50000000, 32, False),
                              EmbeddingConfig(30, [1], 10, 16, False),
                              EmbeddingConfig(30, [1], 1000, 16, False),
                              EmbeddingConfig(5, [1], 10000, 16, False),
                              EmbeddingConfig(5, [1], 100000, 32, False),
                              EmbeddingConfig(27, [1], 4000000, 32, False),
                          ],
                          mlp_sizes=[512, 256, 128],
                          num_numerical_features=10,
                          interact_stride=None)

model_medium = ModelConfig(name="Medium v3",
                           embedding_configs=[
                               EmbeddingConfig(20, [1, 50], 100000, 64, True),
                               EmbeddingConfig(5, [1, 50], 10000000, 64, True),
                               EmbeddingConfig(1, [1, 50], 100000000, 128, True),
                               EmbeddingConfig(1, [1], 100000000, 128, False),
                               EmbeddingConfig(80, [1], 10, 32, False),
                               EmbeddingConfig(60, [1], 1000, 32, False),
                               EmbeddingConfig(80, [1], 100000, 64, False),
                               EmbeddingConfig(24, [1], 200000, 64, False),
                               EmbeddingConfig(40, [1], 10000000, 64, False),
                           ],
                           mlp_sizes=[1024, 512, 256, 128],
                           num_numerical_features=25,
                           interact_stride=7)

model_large = ModelConfig(name="Large v3",
                          embedding_configs=[
                              EmbeddingConfig(40, [1, 100], 100000, 64, True),
                              EmbeddingConfig(16, [1, 100], 15000000, 64, True),
                              EmbeddingConfig(1, [1, 100], 200000000, 128, True),
                              EmbeddingConfig(1, [1], 200000000, 128, False),
                              EmbeddingConfig(100, [1], 10, 32, False),
                              EmbeddingConfig(100, [1], 10000, 32, False),
                              EmbeddingConfig(160, [1], 100000, 64, False),
                              EmbeddingConfig(50, [1], 500000, 64, False),
                              EmbeddingConfig(144, [1], 15000000, 64, False),
                          ],
                          mlp_sizes=[2048, 1024, 512, 256],
                          num_numerical_features=100,
                          interact_stride=8)

model_jumbo = ModelConfig(name="Jumbo v3",
                          embedding_configs=[
                              EmbeddingConfig(50, [1, 200], 100000, 128, True),
                              EmbeddingConfig(24, [1, 200], 20000000, 128, True),
                              EmbeddingConfig(1, [1, 200], 400000000, 256, True),
                              EmbeddingConfig(1, [1], 400000000, 256, False),
                              EmbeddingConfig(100, [1], 10, 32, False),
                              EmbeddingConfig(200, [1], 10000, 64, False),
                              EmbeddingConfig(350, [1], 100000, 128, False),
                              EmbeddingConfig(80, [1], 1000000, 128, False),
                              EmbeddingConfig(216, [1], 20000000, 128, False),
                          ],
                          mlp_sizes=[2048, 1024, 512, 256],
                          num_numerical_features=200,
                          interact_stride=20)

model_colossal = ModelConfig(name="Colossal v3",
                             embedding_configs=[
                                 EmbeddingConfig(100, [1, 300], 100000, 128, True),
                                 EmbeddingConfig(50, [1, 300], 40000000, 256, True),
                                 EmbeddingConfig(1, [1, 300], 2000000000, 256, True),
                                 EmbeddingConfig(1, [1], 1000000000, 256, False),
                                 EmbeddingConfig(100, [1], 10, 32, False),
                                 EmbeddingConfig(400, [1], 10000, 128, False),
                                 EmbeddingConfig(100, [1], 100000, 128, False),
                                 EmbeddingConfig(800, [1], 1000000, 128, False),
                                 EmbeddingConfig(450, [1], 40000000, 256, False),
                             ],
                             mlp_sizes=[4096, 2048, 1024, 512, 256],
                             num_numerical_features=500,
                             interact_stride=30)

model_criteo = ModelConfig(name="Criteo-dlrm-like",
                           embedding_configs=[
                               EmbeddingConfig(26, [1], 100000, 128, False),
                           ],
                           mlp_sizes=[512, 256, 128],
                           num_numerical_features=13,
                           interact_stride=None)

synthetic_models_v3 = {
    "criteo": model_criteo,
    "tiny": model_tiny,
    "small": model_small,
    "medium": model_medium,
    "large": model_large,
    "jumbo": model_jumbo,
    "colossal": model_colossal
}
