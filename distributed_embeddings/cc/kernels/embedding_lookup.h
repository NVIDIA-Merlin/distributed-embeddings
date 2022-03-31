/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef DISTRIBUTED_EMBEDDING_CC_KERNELS_EMBEDDING_LOOKUP_H_
#define DISTRIBUTED_EMBEDDING_CC_KERNELS_EMBEDDING_LOOKUP_H_

#include <string>

namespace tensorflow {
enum class Combiner { Mean = 0, Sum = 1 };
inline Combiner StringToEnum(std::string combiner) {
  return combiner == "mean" ? Combiner::Mean : Combiner::Sum;
}

template <typename Device, typename T, typename Tindices>
struct EmbeddingLookupConstantHotnessFunctor {
  void operator()(const Device& d, T* output_ptr, const T* param_ptr, const Tindices* ids_ptr,
                  Tindices nnz_per_row, Tindices num_rows, Tindices embedding_width,
                  Combiner combiner) const;
};

template <typename Device, typename T, typename Tindices>
struct EmbeddingLookupConstantHotnessGradFunctor {
  void operator()(const Device& d, T* output_ptr, const T* grad_ptr, Tindices nnz_per_row,
                  Tindices num_rows, Tindices embedding_width, Combiner combiner) const;
};

template <typename Device, typename T, typename Tindices>
struct EmbeddingLookupVariableHotnessFunctor {
  void operator()(const Device& d, T* output_ptr, const T* param_ptr, const Tindices* ids_ptr,
                  const Tindices* offsets_ptr, Tindices num_rows, Tindices embedding_width,
                  Combiner combiner) const;
};

template <typename Device, typename T, typename Tindices>
struct EmbeddingLookupVariableHotnessGradFunctor {
  void operator()(const Device& d, T* output_ptr, const T* grad_ptr, const Tindices* offsets_ptr,
                  Tindices num_rows, Tindices embedding_width, Combiner combiner) const;
};

}  // namespace tensorflow

#endif  // DISTRIBUTED_EMBEDDING_CC_KERNELS_EMBEDDING_LOOKUP_H_
