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

#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

REGISTER_OP("ReadVariableNoCopy")
    .Input("resource: resource")
    .Output("value: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      std::vector<shape_inference::ShapeAndType> shape_and_type;
      TF_RETURN_IF_ERROR(shape_inference::ValidateVariableResourceHandle(c, &shape_and_type));
      c->set_output(0, shape_and_type[0].shape);
      return OkStatus();
    });

REGISTER_OP("RowToSplit")
    .Attr("Tindices: {int32, int64}")
    .Input("row_ids: Tindices")
    .Input("shape: int32")
    .Output("row_split: Tindices")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // TODO
      return OkStatus();
    });

REGISTER_OP("EmbeddingLookupVariableHotness")
    .Attr("T: {float}")
    .Attr("Tindices: {int32, int64}")
    .Input("param: T")
    .Input("ids: Tindices")
    .Input("offsets: Tindices")
    .Output("output_params: T")
    .Attr("combiner:  {'sum', 'mean'}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      // vitual input: [m,n], param: [N,p], ids:[nnz], offsets:[m+1]
      // output: [m, p]
      shape_inference::ShapeHandle params_shape;
      shape_inference::ShapeHandle ids_shape;
      shape_inference::ShapeHandle offsets_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &params_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &ids_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &offsets_shape));
      auto outdim_0 = c->Value(c->Dim(offsets_shape, 0));
      // Just in case shape inference happens while batch dim is not None
      if (outdim_0 > 0) {
        outdim_0 -= 1;
      }
      c->set_output(0, c->Matrix(outdim_0, c->Dim(params_shape, 1)));
      return OkStatus();
    });

REGISTER_OP("EmbeddingLookupVariableHotnessGrad")
    .Attr("T: {float}")
    .Attr("Tindices: {int32, int64}")
    .Input("ids: Tindices")
    .Input("offset: Tindices")
    .Input("grad: T")
    .Input("param: T")
    .Output("unique_ids: Tindices")
    .Output("unique_grad: T")
    .Attr("combiner:  {'sum', 'mean'}")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle grad_shape;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &grad_shape));
      c->set_output(0, c->Vector(shape_inference::InferenceContext::kUnknownDim));
      c->set_output(
          1, c->Matrix(shape_inference::InferenceContext::kUnknownDim, c->Dim(grad_shape, 1)));
      return OkStatus();
    });

}  // namespace tensorflow
