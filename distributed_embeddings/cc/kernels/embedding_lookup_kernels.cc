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

#include "embedding_lookup.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/resource_var.h"

namespace tensorflow {

// helper op for sparse read that don't respect copy_on_read_mode
// Compare to sparse_read(resource_gather) method of resource variable,
// as long as a custom sparse read op following this, only difference is earlier lock release
// since copy_on_read_mode result in dense write copying anyway, early unlock won't affect it
class ReadVariableNoCopyOp : public OpKernel {
 public:
  explicit ReadVariableNoCopyOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dtype", &dtype_));
  }
  void Compute(OpKernelContext* context) override {
    core::RefCountPtr<Var> v;
    OP_REQUIRES_OK(context, LookupResource(context, HandleFromInput(context, 0), &v));
    tf_shared_lock ml(*v->mu());
    const Tensor* t = v->tensor();
    OP_REQUIRES(context, dtype_ == t->dtype(),
                errors::InvalidArgument("Trying to read variable(no copy) with wrong dtype."));
    context->set_output(0, *t);
  }

 private:
  DataType dtype_;
};

template <typename Device, typename Tindices>
class RowToSplitOp : public OpKernel {
 public:
  explicit RowToSplitOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // [n, 2]
    const Tensor& row = context->input(0);
    auto num_ids = row.dim_size(0);
    auto num_rows = context->input(1).scalar<int32>()();

    TensorShape output_shape({num_rows + 1});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    RowToSplitFunctor<Device, Tindices>()(context->eigen_device<Device>(),
                                          output->flat<Tindices>().data(),
                                          row.flat<Tindices>().data(), num_ids, num_rows);
  }
};

template <typename Device, typename T, typename Tindices>
class EmbeddingLookupVariableHotnessOp : public OpKernel {
 public:
  explicit EmbeddingLookupVariableHotnessOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &_combiner));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& params = context->input(0);
    const Tensor& ids = context->input(1);
    const Tensor& offsets = context->input(2);

    auto num_rows = offsets.dim_size(0) - 1;
    auto embedding_width = params.dim_size(1);

    auto num_ids = ids.dim_size(0);
    auto ave_red_len = num_ids / num_rows;

    TensorShape output_shape({num_rows, embedding_width});
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    EmbeddingLookupVariableHotnessFunctor<Device, T, Tindices>()(
        context->eigen_device<Device>(), output->flat<T>().data(), params.flat<T>().data(),
        ids.flat<Tindices>().data(), offsets.flat<Tindices>().data(), num_rows, embedding_width,
        StringToEnum(_combiner), ave_red_len);
  }

 private:
  string _combiner;
};

template <typename Device, typename T, typename Tindices>
class EmbeddingLookupVariableHotnessGradOp : public OpKernel {
 public:
  explicit EmbeddingLookupVariableHotnessGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("combiner", &_combiner));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& ids = context->input(0);
    const Tensor& offset_in = context->input(1);
    const Tensor& grad = context->input(2);
    const Tensor& param = context->input(3);
    auto num_ids = ids.dim_size(0);
    auto num_rows = offset_in.dim_size(0) - 1;
    auto embedding_width = grad.dim_size(1);
    auto max_red_len = grad.dim_size(0);
    auto dense_shape_dim0 = param.dim_size(0);

    EmbeddingLookupVariableHotnessGradFunctor<Device, T, Tindices>()(
        context, ids.flat<Tindices>().data(), offset_in.flat<Tindices>().data(),
        grad.flat<T>().data(), num_ids, embedding_width, num_rows, dense_shape_dim0, max_red_len,
        StringToEnum(_combiner));
  }

 private:
  string _combiner;
};

REGISTER_KERNEL_BUILDER(Name("ReadVariableNoCopy").Device(DEVICE_CPU), ReadVariableNoCopyOp);
REGISTER_KERNEL_BUILDER(Name("ReadVariableNoCopy").Device(DEVICE_DEFAULT).HostMemory("resource"),
                        ReadVariableNoCopyOp);
REGISTER_KERNEL_BUILDER(Name("ReadVariableNoCopy").Device(DEVICE_GPU).HostMemory("resource"),
                        ReadVariableNoCopyOp);

#define REGISTER_GPU(T, Tindices)                                                           \
  REGISTER_KERNEL_BUILDER(Name("RowToSplit")                                                \
                              .Device(DEVICE_GPU)                                           \
                              .TypeConstraint<Tindices>("Tindices")                         \
                              .HostMemory("shape"),                                         \
                          RowToSplitOp<Eigen::GpuDevice, Tindices>);                        \
  REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupVariableHotness")                            \
                              .Device(DEVICE_GPU)                                           \
                              .TypeConstraint<T>("T")                                       \
                              .TypeConstraint<Tindices>("Tindices"),                        \
                          EmbeddingLookupVariableHotnessOp<Eigen::GpuDevice, T, Tindices>); \
  REGISTER_KERNEL_BUILDER(Name("EmbeddingLookupVariableHotnessGrad")                        \
                              .Device(DEVICE_GPU)                                           \
                              .TypeConstraint<T>("T")                                       \
                              .TypeConstraint<Tindices>("Tindices"),                        \
                          EmbeddingLookupVariableHotnessGradOp<Eigen::GpuDevice, T, Tindices>);

REGISTER_GPU(float, int64_t)
REGISTER_GPU(float, int32_t)
}  // namespace tensorflow
