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
#define THRUST_IGNORE_CUB_VERSION_CHECK
#define _CG_ABI_EXPERIMENTAL  // enable experimental API
#define ILP 4

#include <cooperative_groups.h>

#include "cub/cub.cuh"
#include "embedding_lookup.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace cg = cooperative_groups;

namespace tensorflow {

template <typename T, typename TIndex, int tile, int row>
__device__ void EmbeddingReduceByIndices(cg::thread_block_tile<tile> g, T* out, const T* params,
                                         int embedding_width, int query_nnz, const TIndex* indices,
                                         TIndex* shmem_indices, Combiner combiner,
                                         const T* weights) {
  T weight = 1;
  int tid = g.thread_rank();
  int row_off = tid / row * row;
  int row_tid = tid % row;
  T result[(row + tile - 1) / tile] = {0};

  // Remainder is handled first
  int remainder = query_nnz % tile;
  // First stage, each CTA load one segment of indices in the sample into shared memory
  g.sync();
  if (tid < remainder) {
    shmem_indices[tid] = indices[tid];
  }
  g.sync();
  // Second stage
  // A CTA first reads indices from shared memory and finds the corresponding entry in the
  // embedding table. Then the CTA reads the embedding vector and accumulates into register file.
  // Each thread in the CTA reads one element of the embedding vector
  _Pragma("unroll 4")
  for (int i = tid / row; i < remainder; i += (tile + row - 1) / row) {
    if (weights != nullptr) weight = weights[shmem_indices[i]];
    _Pragma("unroll")
    for (int j = 0; j < (row + tile - 1) / tile; ++j) {
      if (j * tile + row_tid < embedding_width) {
        result[j] +=
            weight *
            params[shmem_indices[i] * static_cast<int64_t>(embedding_width) + j * tile + row_tid];
      }
    }
  }

  _Pragma("unroll")
  for (int j = 0; j < (row + tile - 1) / tile; ++j) {
    out[j] += result[j];
    result[j] = 0;
  }

  g.sync();
  // Repeat stages above and handle one block size of indices at a time
  for (int processed = remainder; processed < query_nnz; processed += tile) {
    shmem_indices[tid] = indices[processed + tid];
    g.sync();
    _Pragma("unroll 4")
    for (int i = 0; i < row && i < tile; ++i) {
      if (weights != nullptr) weight = weights[shmem_indices[i + row_off]];
      _Pragma("unroll")
      for (int j = 0; j < (row + tile - 1) / tile; ++j) {
        if (j * tile + row_tid < embedding_width) {
          result[j] +=
              weight * params[shmem_indices[i + row_off] * static_cast<int64_t>(embedding_width) +
                              j * tile + row_tid];
        }
      }
    }
    _Pragma("unroll")
    for (int j = 0; j < (row + tile - 1) / tile; ++j) {
      out[j] += result[j];
      result[j] = 0;
    }
    g.sync();
  }

  // reduce down to row elements, only first row have correct result
  for (int i = tile / 2; i >= row; i /= 2) {
    _Pragma("unroll")
    for (int j = 0; j < (row + tile - 1) / tile; ++j) {
      out[j] += g.shfl_down(out[j], i);
    }
  }
}

template <typename T, typename TIndex, int tile, int row>
__device__ void EmbeddingReduceByIndicesWide(cg::thread_block_tile<tile> g, T* out, const T* params,
                                             int embedding_width, int query_nnz,
                                             const TIndex* indices, TIndex* shmem_indices,
                                             Combiner combiner, const T* weights, int rem_width) {
  T weight = 1;
  int tid = g.thread_rank();
  T result[(row + tile - 1) / tile] = {0};

  // Remainder is handled first
  int remainder = query_nnz % tile;
  // First stage, each CTA load one segment of indices in the sample into shared memory
  g.sync();
  if (tid < remainder) {
    shmem_indices[tid] = indices[tid];
  }
  g.sync();
  // Second stage
  // A CTA first reads indices from shared memory and finds the corresponding entry in the
  // embedding table. Then the CTA reads the embedding vector and accumulates into register file.
  // Each thread in the CTA reads one element of the embedding vector
  _Pragma("unroll 4")
  for (int i = 0; i < remainder; ++i) {
    if (weights != nullptr) weight = weights[shmem_indices[i]];
    _Pragma("unroll")
    for (int j = 0; j < (row + tile - 1) / tile; ++j) {
      if (j * tile + tid < rem_width) {
        result[j] +=
            weight *
            params[shmem_indices[i] * static_cast<int64_t>(embedding_width) + j * tile + tid];
      }
    }
  }

  _Pragma("unroll")
  for (int j = 0; j < (row + tile - 1) / tile; ++j) {
    out[j] += result[j];
    result[j] = 0;
  }

  g.sync();
  // Repeat stages above and handle one block size of indices at a time
  for (int processed = remainder; processed < query_nnz; processed += tile) {
    shmem_indices[tid] = indices[processed + tid];
    g.sync();
    _Pragma("unroll 4")
    for (int i = 0; i < tile; ++i) {
      if (weights != nullptr) weight = weights[shmem_indices[i]];
      _Pragma("unroll")
      for (int j = 0; j < (row + tile - 1) / tile; ++j) {
        if (j * tile + tid < rem_width) {
          result[j] +=
              weight *
              params[shmem_indices[i] * static_cast<int64_t>(embedding_width) + j * tile + tid];
        }
      }
    }
    _Pragma("unroll")
    for (int j = 0; j < (row + tile - 1) / tile; ++j) {
      out[j] += result[j];
      result[j] = 0;
    }
    g.sync();
  }
}

template <typename T, typename TIndex, int tile, int row>
__global__ void EmbeddingLookUpVariableHot(const T* params, int embedding_width,
                                           const TIndex* indptr, const TIndex* indices, T* out,
                                           Combiner combiner, TIndex num_rows, const T* weights) {
  auto row_group = cg::tiled_partition<tile>(cg::this_thread_block());

  // smem same size as block size.
  extern __shared__ char shmem[];
  TIndex* shmem_indices = reinterpret_cast<TIndex*>(shmem);
  T* shmem_values = reinterpret_cast<T*>(shmem);
  shmem_indices += threadIdx.y * blockDim.x;

  int num_step = num_rows / gridDim.x;
  indptr += blockIdx.x;
  out += blockIdx.x * embedding_width + threadIdx.x;
  if (blockIdx.x < (num_rows % gridDim.x)) num_step += 1;
  int step_counter = threadIdx.y;
  for (int step = 0; step < num_step; step++) {
    int64_t block_ind_offset = indptr[0];
    int query_nnz = indptr[1] - block_ind_offset;
    // we only want break down skewed long reductions, i.e, power law input backward.
    // These reduction length correlate strongly to batchsize. Let's say we care about perf
    // beyond 1k batchsize in general, then we probably need this threshold <512 to be able
    // to breakdown long reduction in these cases.
    // 128 is chosen so each warp have a full read into indptr when there are 4 of them.
    // it seems works fine, but we can make it a function of launch config if needed
    if (query_nnz > 128 && blockDim.y > 1) {
      T result[(row + tile - 1) / tile] = {0};
      int prev_row_extra =
          (query_nnz % blockDim.y) > threadIdx.y ? threadIdx.y : query_nnz % blockDim.y;
      int row_extra = (query_nnz % blockDim.y) > threadIdx.y ? 1 : 0;
      int row_offset = (query_nnz / blockDim.y) * threadIdx.y + prev_row_extra;
      int row_nnz = (query_nnz / blockDim.y) + row_extra;
      EmbeddingReduceByIndices<T, TIndex, tile, row>(
          row_group, result, params, embedding_width, row_nnz,
          indices + block_ind_offset + row_offset, shmem_indices, combiner, weights);
      __syncthreads();
      _Pragma("unroll")
      for (int j = 0; j < (row + tile - 1) / tile; ++j) {
        shmem_values[threadIdx.y * blockDim.x + threadIdx.x] = result[j];
        __syncthreads();
        if (threadIdx.y == 0) {
          for (int i = 1; i < blockDim.y; i++) {
            result[j] += shmem_values[i * blockDim.x + threadIdx.x];
          }
          if (combiner == Combiner::Mean) {
            result[j] /= query_nnz;
          }
          if (j * tile + threadIdx.x < embedding_width) out[j * tile] = result[j];
        }
        __syncthreads();
      }
    } else {
      // only one row of threads handle one query(output) of embedding
      // the rest of thread can proceed without stucking here
      if (!step_counter) {
        step_counter = blockDim.y;
        T result[(row + tile - 1) / tile] = {0};
        EmbeddingReduceByIndices<T, TIndex, tile, row>(row_group, result, params, embedding_width,
                                                       query_nnz, indices + block_ind_offset,
                                                       shmem_indices, combiner, weights);
        _Pragma("unroll")
        for (int j = 0; j < (row + tile - 1) / tile; ++j) {
          if (combiner == Combiner::Mean) {
            result[j] /= query_nnz;
          }
          if (j * tile + threadIdx.x < embedding_width) out[j * tile] = result[j];
        }
      }
      step_counter -= 1;
    }
    indptr += gridDim.x;
    out += gridDim.x * embedding_width;
  }
}

// version for tile size greater than 32, differences are:
// each tile not within warp so no reduction with shfldown
// need experimental to support tile > 32
// have an outer loop to handle arbitrary embedding_width
template <typename T, typename TIndex, int tile, int row>
__global__ void EmbeddingLookUpVariableHotWide(const T* params, int embedding_width,
                                               const TIndex* indptr, const TIndex* indices, T* out,
                                               Combiner combiner, TIndex num_rows,
                                               const T* weights) {
  // reserve shared memory for thread_block_tile usage.
  __shared__ cg::experimental::block_tile_memory<sizeof(T), tile * 8> shared_for_cg;
  cg::thread_block thb = cg::experimental::this_thread_block(shared_for_cg);
  auto row_group = cg::experimental::tiled_partition<tile>(thb);

  // smem same size as block size.
  extern __shared__ char shmem[];
  TIndex* shmem_indices = reinterpret_cast<TIndex*>(shmem);
  T* shmem_values = reinterpret_cast<T*>(shmem);
  shmem_indices += threadIdx.y * blockDim.x;

  int rem_width = embedding_width;
  for (int out_i = 0; out_i < (embedding_width + row - 1) / row; ++out_i) {
    int cur_id = blockIdx.x;
    while (cur_id < num_rows) {
      TIndex block_ind_offset = indptr[cur_id];
      int query_nnz = indptr[cur_id + 1] - block_ind_offset;
      int64_t block_out_offset = cur_id * embedding_width;
      if (query_nnz > 128 && blockDim.y > 1) {
        T result[(row + tile - 1) / tile] = {0};
        int prev_row_extra =
            (query_nnz % blockDim.y) > threadIdx.y ? threadIdx.y : query_nnz % blockDim.y;
        int row_extra = (query_nnz % blockDim.y) > threadIdx.y ? 1 : 0;
        int row_offset = (query_nnz / blockDim.y) * threadIdx.y + prev_row_extra;
        int row_nnz = (query_nnz / blockDim.y) + row_extra;
        EmbeddingReduceByIndicesWide<T, TIndex, tile, row>(
            row_group, result, params, embedding_width, row_nnz,
            indices + block_ind_offset + row_offset, shmem_indices, combiner, weights, rem_width);
        __syncthreads();

        _Pragma("unroll")
        for (int j = 0; j < (row + tile - 1) / tile; ++j) {
          shmem_values[threadIdx.y * blockDim.x + threadIdx.x] = result[j];
          __syncthreads();
          if (threadIdx.y == 0) {
            for (int i = 1; i < blockDim.y; i++) {
              result[j] += shmem_values[i * blockDim.x + threadIdx.x];
            }
            if (combiner == Combiner::Mean) {
              result[j] /= query_nnz;
            }
            if (j * tile + threadIdx.x < rem_width)
              out[block_out_offset + j * tile + threadIdx.x] = result[j];
          }
          __syncthreads();
        }
      } else {
        // only one row of threads handle one query(output) of embedding
        // the rest of thread can proceed without stucking here
        if ((cur_id / gridDim.x) % blockDim.y == threadIdx.y) {
          T result[(row + tile - 1) / tile] = {0};
          EmbeddingReduceByIndicesWide<T, TIndex, tile, row>(
              row_group, result, params, embedding_width, query_nnz, indices + block_ind_offset,
              shmem_indices, combiner, weights, rem_width);
          _Pragma("unroll")
          for (int j = 0; j < (row + tile - 1) / tile; ++j) {
            if (combiner == Combiner::Mean) {
              result[j] /= query_nnz;
            }
            if (j * tile + threadIdx.x < rem_width)
              out[block_out_offset + j * tile + threadIdx.x] = result[j];
          }
        }
      }
      cur_id += gridDim.x;
    }
    params += row;
    out += row;
    rem_width -= row;
  }
}
template <typename TIndex>
__global__ void RowToSplit(TIndex* split_ptr, const TIndex* row_ptr, TIndex num_ids,
                           TIndex num_rows) {
  // effectively parallel binary search
  auto tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == num_rows) split_ptr[tid] = num_ids;
  if (tid >= num_rows) return;
  TIndex res, begin = 0, end = num_ids - 1;
  while (begin < end) {
    res = (begin + end) / 2;
    if (row_ptr[res * 2] < tid) {
      begin = res + 1;
    } else if (row_ptr[res * 2] > tid) {
      end = res - 1;
    } else {
      end = res;
    }
  }
  split_ptr[tid] = end;
}

template <typename T, typename TIndex>
__global__ void OffsetToWeightsAndRowId(const TIndex* indptr, int32_t* out, T* weights) {
  TIndex block_start_offset = indptr[blockIdx.x];
  TIndex block_end_offset = indptr[blockIdx.x + 1];
  for (TIndex i = block_start_offset + threadIdx.x; i < block_end_offset; i += blockDim.x) {
    out[i] = blockIdx.x;
  }
  if (threadIdx.x == 0 && weights)
    weights[blockIdx.x] = static_cast<T>(1) / static_cast<T>(block_end_offset - block_start_offset);
}

template <typename TIndex>
struct RowToSplitFunctor<Eigen::GpuDevice, TIndex> {
  void operator()(const Eigen::GpuDevice& d, TIndex* split_ptr, const TIndex* row_ptr,
                  TIndex num_ids, TIndex num_rows) const {
    TF_CHECK_OK(GpuLaunchKernel(RowToSplit<TIndex>, num_rows / 512 + 1, 512, 0, d.stream(),
                                split_ptr, row_ptr, num_ids, num_rows));
  }
};

template <typename T, typename TIndex>
struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, T, TIndex> {
  void operator()(const Eigen::GpuDevice& d, T* output_ptr, const T* param_ptr,
                  const TIndex* ids_ptr, const TIndex* offsets_ptr, TIndex num_rows,
                  TIndex embedding_width, Combiner combiner, TIndex ave_red_len) const {
    int next_power_of_two = 1 << Log2Ceiling64(embedding_width);

    // decide number of parallel tile base on reduction length
    int parallel_tile = 1;
    if (ave_red_len >= 256) parallel_tile = 2;
    if (ave_red_len >= 1024) parallel_tile = 4;

    // decide number of threads per tile and adjust number of tile with CUDA limits
    int blockX = next_power_of_two / ILP;
    if (blockX < 32) blockX = 32;
    if (blockX > 256) blockX = 256;
    if (parallel_tile * blockX > 1024) parallel_tile = 1024 / blockX;

    // decide grid dimension and dynamic shared memory size
    dim3 blockDim = dim3(blockX, parallel_tile);
    int smem_size = sizeof(TIndex) > sizeof(T) ? sizeof(TIndex) : sizeof(T);
    smem_size = blockX * parallel_tile * smem_size;
    int gridDim = 32768 / (blockX / 32 * parallel_tile);
    if (gridDim > num_rows) gridDim = num_rows;

    switch (next_power_of_two) {
      case 1:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 1>, gridDim, blockDim,
                                    smem_size, d.stream(), param_ptr, embedding_width, offsets_ptr,
                                    ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 2:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 2>, gridDim, blockDim,
                                    smem_size, d.stream(), param_ptr, embedding_width, offsets_ptr,
                                    ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 4:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 4>, gridDim, blockDim,
                                    smem_size, d.stream(), param_ptr, embedding_width, offsets_ptr,
                                    ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 8:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 8>, gridDim, blockDim,
                                    smem_size, d.stream(), param_ptr, embedding_width, offsets_ptr,
                                    ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 16:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 16>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 32:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 32>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 64:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 64>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 128:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, TIndex, 32, 128>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 256:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHotWide<T, TIndex, 64, 256>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      case 512:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHotWide<T, TIndex, 128, 512>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
      default:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHotWide<T, TIndex, 256, 1024>, gridDim,
                                    blockDim, smem_size, d.stream(), param_ptr, embedding_width,
                                    offsets_ptr, ids_ptr, output_ptr, combiner, num_rows, nullptr));
        break;
    }
  }
};

template <typename T, typename TIndex>
struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, T, TIndex> {
  void operator()(OpKernelContext* context, const TIndex* ids_ptr, const TIndex* offset_in_ptr,
                  const T* grad_ptr, int64_t num_ids, TIndex embedding_width, TIndex num_rows,
                  int64_t dense_shape_dim0, int64_t max_red_len, Combiner combiner) const {
    const auto& cu_stream = GetGpuStream(context);
    cub::CountingInputIterator<int32_t> itr(0);

    // allocate intermediate results buffer
    Tensor tmp_unique_ids;
    Tensor offsets;
    Tensor num_unique_ids;
    Tensor sorted_ids;
    context->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({num_ids}), &tmp_unique_ids);
    context->allocate_temp(DataTypeToEnum<int32_t>::value, TensorShape({num_ids}), &offsets);
    context->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({1}), &num_unique_ids);
    context->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({num_ids}), &sorted_ids);
    auto tmp_unique_ids_ptr = tmp_unique_ids.flat<TIndex>().data();
    auto offsets_ptr = offsets.flat<int32_t>().data();
    auto num_unique_ids_ptr = num_unique_ids.flat<TIndex>().data();
    auto sorted_ids_ptr = sorted_ids.flat<TIndex>().data();

    Tensor row;
    Tensor sorted_row;
    context->allocate_temp(DataTypeToEnum<int32_t>::value, TensorShape({num_ids}), &row);
    context->allocate_temp(DataTypeToEnum<int32_t>::value, TensorShape({num_ids}), &sorted_row);
    auto row_ptr = row.flat<int32_t>().data();
    auto sorted_row_ptr = sorted_row.flat<int32_t>().data();

    T* weights_ptr = nullptr;
    Tensor weights;
    if (combiner == Combiner::Mean) {
      context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({num_rows}), &weights);
      weights_ptr = weights.flat<T>().data();
    }

    TF_CHECK_OK(GpuLaunchKernel(OffsetToWeightsAndRowId<T, TIndex>, num_rows, 32, 0, cu_stream,
                                offset_in_ptr, row_ptr, weights_ptr));

    // Determine temporary device storage requirements
    size_t temp_sort = 0;
    size_t temp_unique = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_sort, ids_ptr, sorted_ids_ptr, row_ptr,
                                    sorted_row_ptr, num_ids, 0, Log2Ceiling64(dense_shape_dim0),
                                    cu_stream);
    cub::DeviceSelect::UniqueByKey(nullptr, temp_unique, sorted_ids_ptr, itr, tmp_unique_ids_ptr,
                                   offsets_ptr, num_unique_ids_ptr, num_ids, cu_stream);
    Tensor temp_storage;
    size_t temp_storage_bytes = temp_sort > temp_unique ? temp_sort : temp_unique;
    context->allocate_temp(DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
                           &temp_storage);

    auto temp_storage_ptr = temp_storage.flat<int8>().data();
    cub::DeviceRadixSort::SortPairs(temp_storage_ptr, temp_sort, ids_ptr, sorted_ids_ptr, row_ptr,
                                    sorted_row_ptr, num_ids, 0, Log2Ceiling64(dense_shape_dim0),
                                    cu_stream);
    cub::DeviceSelect::UniqueByKey(temp_storage_ptr, temp_unique, sorted_ids_ptr, itr,
                                   tmp_unique_ids_ptr, offsets_ptr, num_unique_ids_ptr, num_ids,
                                   cu_stream);

    // copy this back to host. should be ok to sync since there is not much to do in between
    // TF way of doing it seems to be event query base
    TIndex num_unique_ids_host = 0;
    cudaMemcpyAsync(&num_unique_ids_host, num_unique_ids_ptr, sizeof(TIndex),
                    cudaMemcpyDeviceToHost, cu_stream);

    cudaMemcpyAsync(offsets_ptr + num_unique_ids_host, &num_ids, sizeof(int32_t),
                    cudaMemcpyHostToDevice, cu_stream);
    // allocate output
    Tensor* unique_ids = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num_unique_ids_host}), &unique_ids));
    auto unique_ids_ptr = unique_ids->flat<TIndex>().data();
    cudaMemcpyAsync(unique_ids_ptr, tmp_unique_ids_ptr, num_unique_ids_host * sizeof(TIndex),
                    cudaMemcpyDeviceToDevice, cu_stream);

    Tensor* unique_grad = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, TensorShape({num_unique_ids_host, embedding_width}),
                                            &unique_grad));
    auto unique_grad_ptr = unique_grad->flat<T>().data();

    int next_power_of_two = 1 << Log2Ceiling64(embedding_width);

    // decide number of parallel tile base on reduction length
    int parallel_tile = 1;
    if (max_red_len > 512) parallel_tile = 2;
    if (max_red_len > 4096) parallel_tile = 4;
    if (max_red_len > 65536) parallel_tile = 6;

    // decide number of threads per tile and adjust number of tile with CUDA limits
    int blockX = next_power_of_two / ILP;
    if (blockX < 32) blockX = 32;
    if (blockX > 256) blockX = 256;
    if (parallel_tile * blockX > 1024) parallel_tile = 1024 / blockX;

    // decide grid dimension and dynamic shared memory size
    dim3 blockDim = dim3(blockX, parallel_tile);
    int smem_size = sizeof(TIndex) > sizeof(T) ? sizeof(TIndex) : sizeof(T);
    smem_size = blockX * parallel_tile * smem_size;
    int gridDim = 32768 / (blockX / 32 * parallel_tile);
    if (gridDim > num_unique_ids_host) gridDim = num_unique_ids_host;

    switch (next_power_of_two) {
      case 1:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 1>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 2:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 2>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 4:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 4>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 8:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 8>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 16:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 16>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 32:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 32>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 64:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 64>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 128:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHot<T, int32_t, 32, 128>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 256:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHotWide<T, int32_t, 64, 256>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      case 512:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHotWide<T, int32_t, 128, 512>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
      default:
        TF_CHECK_OK(GpuLaunchKernel(EmbeddingLookUpVariableHotWide<T, int32_t, 256, 1024>, gridDim,
                                    blockDim, smem_size, cu_stream, grad_ptr, embedding_width,
                                    offsets_ptr, sorted_row_ptr, unique_grad_ptr, Combiner::Sum,
                                    num_unique_ids_host, weights_ptr));
        break;
    }
  }
};

template struct RowToSplitFunctor<Eigen::GpuDevice, int64_t>;
template struct RowToSplitFunctor<Eigen::GpuDevice, int32_t>;
template struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, float, int32_t>;
template struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, float, int32_t>;

}  // namespace tensorflow
