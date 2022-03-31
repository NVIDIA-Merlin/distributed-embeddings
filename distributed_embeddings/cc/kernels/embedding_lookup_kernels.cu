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

#define EIGEN_USE_GPU

#include "embedding_lookup.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

template <typename T, typename Tindices>
__device__ void EmbeddingReduceByIndices(T* out, const T* params, Tindices embedding_width,
                                         Tindices query_nnz, const Tindices* indices,
                                         Tindices* shmem_indices, Combiner combiner) {
  int tid = threadIdx.x;
  T result = 0;

  // Remainder is handled first
  int remainder = query_nnz % blockDim.x;
  // First stage, each CTA load one segment of indices in the sample into shared memory
  if (tid < remainder) {
    shmem_indices[tid] = indices[tid];
  }
  __syncthreads();
  // Second stage
  // A CTA first reads indices from shared memory and finds the corresponding entry in the embedding
  // table. Then the CTA reads the embedding vector and accumulates into register file. Each thread
  // in the CTA reads one element of the embedding vector
#pragma unroll 4
  for (int i = 0; i < remainder; ++i) {
    result += params[shmem_indices[i] * static_cast<int64_t>(embedding_width) + tid];
  }

  // Repeat stages above and handle one block size of indices at a time
  for (int processed = remainder; processed < query_nnz; processed += blockDim.x) {
    shmem_indices[tid] = indices[processed + tid];
    __syncthreads();

#pragma unroll 4
    for (int i = 0; i < blockDim.x; ++i) {
      result += params[shmem_indices[i] * static_cast<int64_t>(embedding_width) + tid];
    }
  }

  if (combiner == Combiner::Mean) {
    result /= query_nnz;
  }

  out[tid] = result;
}

template <typename T, typename Tindices>
__device__ void EmbeddingExpandGrad(T* value, const T* grad, Tindices embedding_width,
                                    Tindices query_nnz, Combiner combiner) {
  int tid = threadIdx.x;

  T g = grad[tid];
  if (combiner == Combiner::Mean) {
    g /= query_nnz;
  }

#pragma unroll 4
  for (int i = 0; i < query_nnz; ++i) {
    value[i * static_cast<int64_t>(embedding_width) + tid] = g;
  }
}

template <typename T, typename Tindices>
__global__ void EmbeddingLookUpConstantHot(const T* params, Tindices embedding_width,
                                           Tindices query_nnz, const Tindices* indices, T* out,
                                           Combiner combiner) {
  int64_t block_ind_offset = blockIdx.x * query_nnz;
  int64_t block_out_offset = blockIdx.x * embedding_width;
  // smem same size as block size.
  extern __shared__ char shmem[];
  Tindices* shmem_indices = reinterpret_cast<Tindices*>(shmem);
  // each block handle one query(output) of embedding
  EmbeddingReduceByIndices(out + block_out_offset, params, embedding_width, query_nnz,
                           indices + block_ind_offset, shmem_indices, combiner);
}

template <typename T, typename Tindices>
__global__ void EmbeddingLookUpGradConstantHot(const T* grad, Tindices embedding_width,
                                               Tindices query_nnz, T* value, Combiner combiner) {
  int64_t block_value_offset = blockIdx.x * query_nnz * embedding_width;
  int64_t block_grad_offset = blockIdx.x * embedding_width;
  EmbeddingExpandGrad(value + block_value_offset, grad + block_grad_offset, embedding_width,
                      query_nnz, combiner);
}

template <typename T, typename Tindices>
__global__ void EmbeddingLookUpVariableHot(const T* params, Tindices embedding_width,
                                           const Tindices* indptr, const Tindices* indices, T* out,
                                           Combiner combiner) {
  Tindices block_ind_offset = indptr[blockIdx.x];
  Tindices query_nnz = indptr[blockIdx.x + 1] - block_ind_offset;
  int64_t block_out_offset = blockIdx.x * embedding_width;
  // smem same size as block size.
  extern __shared__ char shmem[];
  Tindices* shmem_indices = reinterpret_cast<Tindices*>(shmem);
  // each block handle one query(output) of embedding
  EmbeddingReduceByIndices(out + block_out_offset, params, embedding_width, query_nnz,
                           indices + block_ind_offset, shmem_indices, combiner);
}

template <typename T, typename Tindices>
__global__ void EmbeddingLookUpGradVariableHot(const T* grad, Tindices embedding_width,
                                               const Tindices* indptr, T* value,
                                               Combiner combiner) {
  Tindices block_ind_offset = indptr[blockIdx.x];
  Tindices query_nnz = indptr[blockIdx.x + 1] - block_ind_offset;
  int64_t block_value_offset = block_ind_offset * embedding_width;
  int64_t block_grad_offset = blockIdx.x * embedding_width;
  EmbeddingExpandGrad(value + block_value_offset, grad + block_grad_offset, embedding_width,
                      query_nnz, combiner);
}

template <typename T, typename Tindices>
struct EmbeddingLookupConstantHotnessFunctor<Eigen::GpuDevice, T, Tindices> {
  void operator()(const Eigen::GpuDevice& d, T* output_ptr, const T* param_ptr,
                  const Tindices* ids_ptr, Tindices nnz_per_row, Tindices num_rows,
                  Tindices embedding_width, Combiner combiner) const {
    TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpConstantHot<T, Tindices>, num_rows, embedding_width,
                                embedding_width * sizeof(Tindices), d.stream(), param_ptr,
                                embedding_width, nnz_per_row, ids_ptr, output_ptr, combiner));
  }
};

template <typename T, typename Tindices>
struct EmbeddingLookupConstantHotnessGradFunctor<Eigen::GpuDevice, T, Tindices> {
  void operator()(const Eigen::GpuDevice& d, T* output_ptr, const T* grad_ptr, Tindices nnz_per_row,
                  Tindices num_rows, Tindices embedding_width, Combiner combiner) const {
    TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpGradConstantHot<T, Tindices>, num_rows,
                                embedding_width, 0, d.stream(), grad_ptr, embedding_width,
                                nnz_per_row, output_ptr, combiner));
  }
};

template <typename T, typename Tindices>
struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, T, Tindices> {
  void operator()(const Eigen::GpuDevice& d, T* output_ptr, const T* param_ptr,
                  const Tindices* ids_ptr, const Tindices* offsets_ptr, Tindices num_rows,
                  Tindices embedding_width, Combiner combiner) const {
    TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, Tindices>, num_rows, embedding_width,
                                embedding_width * sizeof(Tindices), d.stream(), param_ptr,
                                embedding_width, offsets_ptr, ids_ptr, output_ptr, combiner));
  }
};

template <typename T, typename Tindices>
struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, T, Tindices> {
  void operator()(const Eigen::GpuDevice& d, T* output_ptr, const T* grad_ptr,
                  const Tindices* offsets_ptr, Tindices num_rows, Tindices embedding_width,
                  Combiner combiner) const {
    TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpGradVariableHot<T, Tindices>, num_rows,
                                embedding_width, 0, d.stream(), grad_ptr, embedding_width,
                                offsets_ptr, output_ptr, combiner));
  }
};

template struct EmbeddingLookupConstantHotnessFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupConstantHotnessGradFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupConstantHotnessFunctor<Eigen::GpuDevice, float, int32_t>;
template struct EmbeddingLookupConstantHotnessGradFunctor<Eigen::GpuDevice, float, int32_t>;
template struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, float, int32_t>;
template struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, float, int32_t>;

}  // namespace tensorflow
