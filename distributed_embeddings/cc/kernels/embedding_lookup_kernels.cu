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
#include "cuco/static_map.cuh"
#include "embedding_lookup.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace cg = cooperative_groups;

namespace tensorflow {

constexpr int kWarpSize = 32;
constexpr int kReduceIndicesPerWarp = 32;
constexpr int kWarpsPerCTA = 4;

template<typename IndexT>
__global__ void FlagWgradBoundaries(int32_t* flags,
                                    const IndexT* __restrict sorted_indices,
                                    IndexT num_indices) {
  IndexT idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    flags[0] = 0;
  }

  for (; idx < num_indices - 1; idx += blockDim.x * gridDim.x) {
    IndexT curr_id = sorted_indices[idx];
    IndexT next_id = sorted_indices[idx + 1];
    flags[idx + 1] = static_cast<int32_t>(next_id != curr_id);
  }
}

/**
 * @brief Performs reduce within each warp. If the gradients for a particular category only span 1 warp, or the warp is
 * the last one accumulating gradients for that category, then it outputs the full/partial wgrad to unique_wgrad.
 * Otherwise, if the gradients span across multiple warps, those warps will store this partial wgrad in partial_wgrad.
 *
 * @tparam T Gradient type
 * @tparam IndexT Indices Type
 * @tparam row_num_accumulators Equal to round_up(embedding_width/warpSize)
 * @param grad The dY gradients
 * @param embedding_width The number of elements in each embedding vector
 * @param indptr Row offsets, num rows to reduce is computed by indptr[n+1] - indptr[n]
 * @param sorted_indices Category Ids corresponding to each row
 * @param sorted_row_ids Gradient Ids, i.e indices into dY
 * @param unique_wgrad_ids Ids into unique_wgrad, since unique_wgrad is sparse
 * @param unique_wgrad The output wgrad buffer containing fully reduced gradients
 * @param partial_wgrad Intermediate buffer to store partially reduced gradients for each warp
 * @param combiner How to reduce the gradients dY
 * @param num_unique_ids The number of unique categories/embeddings to compute wgrad for
 * @param weights Used for mean reduce
 */
template <typename T, typename IndexT, int row_num_accumulators>
__global__ void EmbeddingWgradPartialReduceVariableHot(const T* grad,
                                                       int embedding_width,
                                                       const int32_t* indptr,
                                                       const IndexT* sorted_indices,
                                                       const int32_t* sorted_row_ids,
                                                       const int32_t* unique_wgrad_ids,
                                                       T* unique_wgrad,
                                                       T* partial_wgrad,
                                                       Combiner combiner,
                                                       IndexT num_unique_ids,
                                                       const T* weights) {
  T acc[row_num_accumulators] = { 0 };

  // thread positional data
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;

  IndexT start_index = (blockIdx.x * num_warps + warp_id) * kReduceIndicesPerWarp;
  IndexT num_indices = indptr[num_unique_ids];

  if (start_index >= num_indices) {
    return;
  }

  int num_local_indices = min(kReduceIndicesPerWarp, static_cast<int>(num_indices - start_index));

  __shared__ int32_t s_dY_id[kWarpsPerCTA][kReduceIndicesPerWarp];
  __shared__ T s_weights[kWarpsPerCTA][kReduceIndicesPerWarp];
  __shared__ IndexT s_unique_wgrad_ids[kWarpsPerCTA][kReduceIndicesPerWarp];
  __shared__ int s_write_wgrad[kWarpsPerCTA][kReduceIndicesPerWarp+1];

  sorted_row_ids += start_index;
  sorted_indices += start_index;
  unique_wgrad_ids += start_index;

  for (int i = lane_id; i < num_local_indices; i += kWarpSize) {
    s_dY_id[warp_id][i] = sorted_row_ids[i];
    s_unique_wgrad_ids[warp_id][i] = unique_wgrad_ids[i];
    s_write_wgrad[warp_id][i] = sorted_indices[i] != sorted_indices[i + 1];
  }

  // Whether this warp should output partial wgrad
  if (lane_id == 0) {
    s_write_wgrad[warp_id][num_local_indices] = sorted_indices[num_local_indices - 1] == sorted_indices[num_local_indices];
  }

  if (weights) {
    for (int i = lane_id; i < num_local_indices; i += kWarpSize) {
      s_weights[warp_id][i] = weights[s_dY_id[warp_id][i]];
    }
  } else {
    for (int i = lane_id; i < num_local_indices; i += kWarpSize) {
      s_weights[warp_id][i] = 1;
    }
  }

  __syncwarp();

  for (int i = 0; i < num_local_indices; ++i) {

    IndexT dY_id = s_dY_id[warp_id][i];
    T weight = s_weights[warp_id][i];

    // load dY
    const auto src = &grad[dY_id * embedding_width];
    _Pragma("unroll")
    for (int j = 0; j < row_num_accumulators; ++j) {
      if (j * kWarpSize + lane_id < embedding_width) {
        acc[j] += weight * src[j * kWarpSize + lane_id];
      }
    }

    // when category changes, write out to wgrad
    if (s_write_wgrad[warp_id][i]) {
      IndexT unique_row_id = s_unique_wgrad_ids[warp_id][i];
      auto dst = &unique_wgrad[unique_row_id * embedding_width];

      _Pragma("unroll")
      for (int j = 0; j < row_num_accumulators; ++j) {
        if (j * kWarpSize + lane_id < embedding_width) {
          dst[j * kWarpSize + lane_id] = acc[j];
        }
        acc[j] = 0.f; // reset accumulator
      }
    }
  }

  // If the number of dY gradients spanned more than this warp for the same category
  // then we need to write out the partial reduce to the partial wgrad buffer
  if (start_index + num_local_indices < num_indices) {
    // category == next_category
    if (s_write_wgrad[warp_id][num_local_indices]) {
      auto dst = &partial_wgrad[(blockIdx.x * num_warps + warp_id) * embedding_width];

      _Pragma("unroll")
      for (int j = 0; j < row_num_accumulators; ++j) {
        if (j * kWarpSize + lane_id < embedding_width) {
          dst[j * kWarpSize + lane_id] = acc[j];
        }
      }
    }
  }
}

/**
 * @brief Reduces the partial wgrads and outputs the final reduction to unique_wgrad. Each warp will reduce N many
 * partial wgrads, and then atomically add this local reduction to the output unique_wgrad buffer.
 *
 * @tparam T Gradient type
 * @tparam IndexT Indices Type
 * @tparam row_num_accumulators Equal to round_up(embedding_width/warpSize)
 * @param embedding_width The number of elements in each embedding vector
 * @param indptr Row offsets, num rows to reduce is computed by indptr[n+1] - indptr[n]
 * @param sorted_indices Category Ids corresponding to each row
 * @param unique_wgrad_ids Ids into unique_wgrad, since unique_wgrad is sparse
 * @param unique_wgrad The output wgrad buffer containing fully reduced gradients
 * @param partial_wgrad Intermediate buffer to store partially reduced gradients for each warp
 * @param combiner How to reduce the gradients dY
 * @param num_unique_ids The number of unique categories/embeddings to compute wgrad for
 */
template <typename T, typename IndexT, int row_num_accumulators>
__global__ void EmbeddingWgradFinalReduceVariableHot(int embedding_width,
                                                     const int32_t* indptr,
                                                     const IndexT* sorted_indices,
                                                     const int32_t* unique_wgrad_ids,
                                                     T* unique_wgrad,
                                                     T* partial_wgrad,
                                                     Combiner combiner,
                                                     IndexT num_unique_ids) {
  T acc[row_num_accumulators] = { 0 };

  // thread positional data
  const int lane_id = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int num_warps = blockDim.x >> 5;

  IndexT start_index = (blockIdx.x * num_warps + warp_id) * kReduceIndicesPerWarp;

  IndexT num_indices = indptr[num_unique_ids];
  // round down because the last warp writes to unique_wgrad, so we only need to accumulate for all
  // warps but the last warp
  IndexT num_partial_wgrads = num_indices / kReduceIndicesPerWarp;

  if (start_index >= num_indices) {
    return;
  }

  int num_local_partial_wgrads = min(kReduceIndicesPerWarp, static_cast<int>(num_partial_wgrads - start_index));

  __shared__ IndexT s_unique_row_ids[kWarpsPerCTA][kReduceIndicesPerWarp];
  __shared__ int s_has_partial_wgrad[kWarpsPerCTA][kReduceIndicesPerWarp];
  __shared__ int s_write_wgrad[kWarpsPerCTA][kReduceIndicesPerWarp];

  for (int i = lane_id; i < num_local_partial_wgrads; i += kWarpSize) {
    IndexT next_warp_start = (start_index + i + 1) * kReduceIndicesPerWarp;
    IndexT next_next_warp_start = (start_index + i + 2) * kReduceIndicesPerWarp;

    auto warp_last_category = sorted_indices[next_warp_start - 1];
    auto next_warp_first_category = sorted_indices[next_warp_start];

    s_unique_row_ids[warp_id][i] = unique_wgrad_ids[next_warp_start - 1];
    s_has_partial_wgrad[warp_id][i] = warp_last_category == next_warp_first_category;

    // Precompute where to add accumulated wgrad to unqiue_wgrad in order to decrease
    // latency on critical path and improve memory bandwidth.
    if (s_has_partial_wgrad[warp_id][i] && next_next_warp_start < num_indices) {
      // In the example below, there is no partial wgrad for Warp N+2 because categories B
      // do not span to the next warp.
      //
      ///         Warp N          Warp N+1        Warp N+2         Warp N+3
      //    --------------------------------------------------------------------
      //    |      Cat A    ||      Cat A    || Cat A | Cat B ||    Cat C      |  = sorted_indices
      //    --------------------------------------------------------------------
      //    |    PartWgrad  ||    PartWgrad  ||      N/A      ||   PartWgrad   |  = partial_wgrad
      //    --------------------------------------------------------------------
      //
      // To determine whether we should flush our accumulated wgrad to gmem (unique_wgrad)
      // we need to look 2 warps ahead. If the first category of the next-next warp does not
      // match, then the next warp has already written its accumulated wgrad to unique_wgrad
      // meaning our current warp contains the last partial wgrad so we need to write it.
      // The other condition for writing is if we are the last loaded wgrad in the warp
      IndexT next_next_warp_first_category = sorted_indices[next_next_warp_start];
      s_write_wgrad[warp_id][i] = (warp_last_category != next_next_warp_first_category) || (i == num_local_partial_wgrads - 1);
    } else {
      s_write_wgrad[warp_id][i] = s_has_partial_wgrad[warp_id][i];
    }
  }

  __syncwarp();

  for (int i = 0; i < num_local_partial_wgrads; ++i) {

    bool have_partial_wgrad = s_has_partial_wgrad[warp_id][i];
    bool write_wgrad = s_write_wgrad[warp_id][i];

    // accumulate partial wgrad
    if (have_partial_wgrad) {
      IndexT dY_id = start_index + i;
      const auto src = &partial_wgrad[dY_id * embedding_width];

#pragma unroll
      for (int j = 0; j < row_num_accumulators; ++j) {
        if (j * kWarpSize + lane_id < embedding_width) {
          acc[j] += src[j * kWarpSize + lane_id];
        }
      }
    }

    if (write_wgrad) {
      IndexT unique_row_id = s_unique_row_ids[warp_id][i];
      auto dst = &unique_wgrad[unique_row_id * embedding_width];

      _Pragma("unroll")
      for (int j = 0; j < row_num_accumulators; ++j) {
        if (j * kWarpSize + lane_id < embedding_width) {
          atomicAdd(&dst[j * kWarpSize + lane_id], acc[j]);
        }
        acc[j] = 0.f; // reset accumulator
      }
    }
  }
}

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
__global__ void EmbeddingLookUpVariableHot(const T* __restrict params, int embedding_width,
                                           const TIndex* __restrict indptr, const TIndex* __restrict indices, T* __restrict out,
                                           Combiner combiner, TIndex num_rows, const T* __restrict weights) {
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
          if (combiner == Combiner::Mean && query_nnz > 0) {
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
          if (combiner == Combiner::Mean && query_nnz > 0) {
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
// have an outer loop to handle arbitrary embedding_width
template <typename T, typename TIndex, int tile, int row>
__global__ void EmbeddingLookUpVariableHotWide(const T* params, int embedding_width,
                                               const TIndex* indptr, const TIndex* indices, T* out,
                                               Combiner combiner, TIndex num_rows,
                                               const T* weights) {
#if __CUDACC_VER_MAJOR__ >= 12
  // According to cuda doc, on compute capability 80 or higher, this should consume no memory
  __shared__ cg::block_tile_memory<tile * 8> shared_for_cg;
  cg::thread_block thb = cg::this_thread_block(shared_for_cg);
  auto row_group = cg::tiled_partition<tile>(thb);
#else
  // unchanged legacy code. these are under experimental namespace before cuda 12.0
  __shared__ cg::experimental::block_tile_memory<sizeof(T), tile * 8> shared_for_cg;
  cg::thread_block thb = cg::experimental::this_thread_block(shared_for_cg);
  auto row_group = cg::experimental::tiled_partition<tile>(thb);
#endif

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
            if (combiner == Combiner::Mean && query_nnz > 0) {
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
            if (combiner == Combiner::Mean && query_nnz > 0) {
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

// The kernel does following things:
// - generate available indices from count array
// - try insert new value from available indices with each key
// - insert either succeed, or get existed value from pervious batch/other parallel threads
// - now we have needed output, update count array for future available index generation
template <typename ViewT, typename T, typename CountT>
__global__ void SearchAndUpdate(ViewT view, const T* keys, T* values, T* avails, CountT* counts,
                                T num_elem, int* g_counter, T capacity) {
  cg::grid_group grid = cg::this_grid();
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // set global atomic counters to save a memset outside
  g_counter[0] = 0;
  g_counter[1] = 0;
  grid.sync();

  // Find at most num_elem indices where count is zero
  // TODO: maybe randomize where to start and avoid slowdown when half full?
  CountT count;
  int avail_offset;
  for (int i = tid; i < capacity; i += blockDim.x * gridDim.x) {
    count = counts[i];
    if (0 == count) {
      avail_offset = atomicAdd(g_counter, 1);
      if (avail_offset >= num_elem) break;
      avails[avail_offset] = i;
    }
  }
  grid.sync();

  // now we have available indices, try insert them with keys
  int num_avail = g_counter[0];
  T key, value;
  // First deal with case where we still have empty slot but not enough to do in one go
  if (num_avail > 0 && num_avail < num_elem) {
    if (tid < num_avail) {
      int cur_offset = atomicAdd(g_counter + 1, 1);
      value = avails[tid];
      while (cur_offset < num_elem) {
        key = keys[cur_offset];
#if __CUDA_ARCH__ < 700
        if constexpr (cuco::detail::is_packable<ViewT::value_type>()) {
#endif
          auto [iter, inserted] = view.insert_and_find(cuco::make_pair(key, value));
          counts[iter->second] += 1;
          values[cur_offset] = iter->second;
          if (inserted) break;
#if __CUDA_ARCH__ < 700
          // TODO(deyuf): add fallback logic determinism and pre-volta gpu. might need multi-kernel
        }
#endif
        cur_offset = atomicAdd(g_counter + 1, 1);
      }
    }
    // above run could stop before checking all keys, when all avaiable indices are inserted
    grid.sync();

    // threads with tid>g_counter will continue and insert_and_find remaining keys with default
    // g_counter >= num_elem means all thread should returns since all keys are looked up already
    if (tid < g_counter[1]) return;
  }
  // drop rest of no longer needed threads after possible grid sync
  if (tid >= num_elem) return;

  // Three cases we end up here:
  // - we have enough new indices for use in one go, all num_elem threads got here
  // - there is no new indices to use at all, all num_elem threads got here
  // - we run out of new indices during above if, only tid matching never looked up key got here
  key = keys[tid];

  // Don't insert OOV keys so table remain not full
  if (num_avail >= num_elem) {
    value = avails[tid];
#if __CUDA_ARCH__ < 700
    if constexpr (cuco::detail::is_packable<ViewT::value_type>()) {
#endif
      auto [iter, inserted] = view.insert_and_find(cuco::make_pair(key, value));
      if (!inserted) value = iter->second;
#if __CUDA_ARCH__ < 700
      // TODO(deyuf): need fallback since pre-volta gpu will fail.
    }
#endif
  } else {
    value = 0;
    auto s_view = typename cuco::static_map<T, T>::device_view(view);
    auto found = s_view.find(key);
    if (found != s_view.end()) value = found->second;
  }
  // update count so this index won't be used again
  atomicAdd(counts + value, (CountT)1);
  // write out to output
  values[tid] = value;
}

template <typename T, typename CountT>
struct IntegerLookupFunctor<Eigen::GpuDevice, T, CountT> {
  void operator()(OpKernelContext* context, T* table_ptr, CountT* count_ptr, const T* keys_ptr,
                  T* value_ptr, T num_elem, bool init, int64_t capacity) const {
    const auto& cu_stream = GetGpuStream(context);

    // get a mutable view from TF managed memory, initialize if needed
    auto table_capacity = capacity * 3 / 2;
    T constexpr empty_key_sentinel = -1;
    T constexpr empty_value_sentinel = -1;
    auto slot = reinterpret_cast<typename cuco::static_map<T, T>::pair_atomic_type*>(table_ptr);
    if (init) {
      using atomic_key_type = typename cuco::static_map<T, T>::atomic_key_type;
      using atomic_mapped_type = typename cuco::static_map<T, T>::atomic_mapped_type;
      auto grid_size = (table_capacity + 1023) / 1024;
      cuco::detail::initialize<256, atomic_key_type, atomic_mapped_type>
          <<<grid_size, 256, 0, cu_stream>>>(slot, cuco::empty_key{empty_key_sentinel},
                                             cuco::empty_value{empty_value_sentinel},
                                             table_capacity);
    }
    auto view = typename cuco::static_map<T, T>::device_mutable_view(
        slot, table_capacity, cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel});

    // counters to figure out offsets between threads
    Tensor atomic_counter;
    context->allocate_temp(DT_INT32, TensorShape({static_cast<int64_t>(2)}), &atomic_counter);
    auto atomic_counter_ptr = atomic_counter.flat<int>().data();
    // DRAM workspace buffer to store new indices available for use
    Tensor temp_avail;
    context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({static_cast<int64_t>(num_elem)}),
                           &temp_avail);
    auto temp_avail_ptr = temp_avail.flat<int64_t>().data();

    int num_threads = 512;
    // TODO: add loop for batch dim and get device prop from TF to figure safe/largest num_blocks
    // For now, use max(enough_for_batch, 64) since most cards we care have more than 64 sm
    auto num_blocks = (num_elem + num_threads - 1) / num_threads;
    num_blocks = num_blocks < 64 ? 64 : num_blocks;
    void* args[] = {&view,      &keys_ptr, &value_ptr,          &temp_avail_ptr,
                    &count_ptr, &num_elem, &atomic_counter_ptr, &capacity};
    cudaLaunchCooperativeKernel(
        (void*)SearchAndUpdate<typename cuco::static_map<T, T>::device_mutable_view, T, CountT>,
        num_blocks, num_threads, args, 0, cu_stream);
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

    int64_t num_partial_wgrad_rows = std::ceil((float)num_ids / kReduceIndicesPerWarp);

    // allocate intermediate results buffer
    Tensor tmp_unique_ids;
    Tensor offsets;
    Tensor num_unique_ids;
    Tensor sorted_ids;
    Tensor tmp_wgrad_boundaries;
    Tensor tmp_unqiue_wgrad_ids;
    Tensor tmp_partial_wgrad;
    context->allocate_temp(DataTypeToEnum<int32_t>::value, TensorShape({num_ids}), &tmp_wgrad_boundaries);
    context->allocate_temp(DataTypeToEnum<int32_t>::value, TensorShape({num_ids}), &tmp_unqiue_wgrad_ids);
    context->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({num_ids}), &tmp_unique_ids);
    context->allocate_temp(DataTypeToEnum<int32_t>::value, TensorShape({num_ids}), &offsets);
    context->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({1}), &num_unique_ids);
    context->allocate_temp(DataTypeToEnum<TIndex>::value, TensorShape({num_ids}), &sorted_ids);
    context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({num_partial_wgrad_rows, embedding_width}), &tmp_partial_wgrad);
    auto tmp_wgrad_boundaries_ptr = tmp_wgrad_boundaries.flat<int32_t>().data();
    auto tmp_unique_wgrad_ids_ptr = tmp_unqiue_wgrad_ids.flat<int32_t>().data();
    auto tmp_partial_wgrad_ptr = tmp_partial_wgrad.flat<T>().data();
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
    size_t temp_scan = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_sort, ids_ptr, sorted_ids_ptr, row_ptr,
                                    sorted_row_ptr, num_ids, 0, Log2Ceiling64(dense_shape_dim0),
                                    cu_stream);
    cub::DeviceSelect::UniqueByKey(nullptr, temp_unique, sorted_ids_ptr, itr, tmp_unique_ids_ptr,
                                   offsets_ptr, num_unique_ids_ptr, num_ids, cu_stream);
    cub::DeviceScan::InclusiveSum(nullptr, temp_scan, tmp_wgrad_boundaries_ptr, tmp_unique_wgrad_ids_ptr,
                                    num_ids, cu_stream);
    Tensor temp_storage;
    size_t temp_storage_bytes = std::max(temp_sort, std::max(temp_unique, temp_scan));
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

    dim3 dim_block_flag(256);
    dim3 dim_grid_flag(std::ceil((float)num_ids / dim_block_flag.x));
    TF_CHECK_OK(GpuLaunchKernel(FlagWgradBoundaries<TIndex>, dim_grid_flag, dim_block_flag, 0, cu_stream,
                                tmp_wgrad_boundaries_ptr, sorted_ids_ptr, num_ids));

    cub::DeviceScan::InclusiveSum(temp_storage_ptr, temp_scan, tmp_wgrad_boundaries_ptr,
                                  tmp_unique_wgrad_ids_ptr, num_ids, cu_stream);

    dim3 dim_block(kWarpSize * kWarpsPerCTA);
    dim3 dim_grid_partial(std::ceil((float)num_ids / dim_block.x));
    dim3 dim_grid_final(std::ceil((float)num_partial_wgrad_rows / dim_block.x));

  #define LAUNCH_WGRAD_KERNELS(num_accumulators) \
    TF_CHECK_OK(GpuLaunchKernel(EmbeddingWgradPartialReduceVariableHot<T, TIndex, num_accumulators>, dim_grid_partial, \
                                dim_block, 0, cu_stream, grad_ptr, embedding_width, \
                                offsets_ptr, sorted_ids_ptr, sorted_row_ptr, tmp_unique_wgrad_ids_ptr, \
                                unique_grad_ptr, tmp_partial_wgrad_ptr, Combiner::Sum, \
                                num_unique_ids_host, weights_ptr)); \
    TF_CHECK_OK(GpuLaunchKernel(EmbeddingWgradFinalReduceVariableHot<T, TIndex, num_accumulators>, dim_grid_final, \
                                dim_block, 0, cu_stream, embedding_width, \
                                offsets_ptr, sorted_ids_ptr, tmp_unique_wgrad_ids_ptr, \
                                unique_grad_ptr, tmp_partial_wgrad_ptr, Combiner::Sum, \
                                num_unique_ids_host));                            

    if (embedding_width <= 32) {
      LAUNCH_WGRAD_KERNELS(1);
    } else if (embedding_width <= 64) {
      LAUNCH_WGRAD_KERNELS(2);
    } else if (embedding_width <= 128) {
      LAUNCH_WGRAD_KERNELS(4);
    } else if (embedding_width <= 256) {
      LAUNCH_WGRAD_KERNELS(8);
    } else if (embedding_width <= 512) {
      LAUNCH_WGRAD_KERNELS(16);
    } else if (embedding_width <= 1024) {
      LAUNCH_WGRAD_KERNELS(32);
    } else {
      throw std::runtime_error("Embedding width > 1024 not currently supported");
    }
  }
};

/*
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
};*/

template struct RowToSplitFunctor<Eigen::GpuDevice, int64_t>;
template struct RowToSplitFunctor<Eigen::GpuDevice, int32_t>;
template struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupVariableHotnessFunctor<Eigen::GpuDevice, float, int32_t>;
template struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, float, int64_t>;
template struct EmbeddingLookupVariableHotnessGradFunctor<Eigen::GpuDevice, float, int32_t>;
template struct IntegerLookupFunctor<Eigen::GpuDevice, int64_t, uint32_t>;

}  // namespace tensorflow
