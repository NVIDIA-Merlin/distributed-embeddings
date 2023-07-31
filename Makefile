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

CXX := g++
CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc
PYTHON_BIN_PATH = python

CUDA_VERSION := $(shell nvcc -V | grep release | grep -Eo '[+-]?[0-9]+([.][0-9]+)?' | head -1)
CUDA_GENCODE = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86
ifeq ($(shell expr $(CUDA_VERSION) ">=" "11.8"), 1)
	  CUDA_GENCODE += -gencode arch=compute_90,code=sm_90
	endif

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_VERSION := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(int(tf.__version__.split(".")[1]))')
ifeq ($(shell expr $(TF_VERSION) \>= 10), 1)
	  CPP_STD := 17
	else
	  CPP_STD := 14
	endif

CFLAGS = ${TF_CFLAGS} -O3 -std=c++${CPP_STD}
LDFLAGS = -shared ${TF_LFLAGS}

SRC = embedding_lookup_kernels

CXX_OBJS = $(SRC:%=%.cc.o)
NVCC_OBJS = $(SRC:%=%.cu.o)

TARGET_LIB = distributed_embeddings/python/ops/_embedding_lookup_ops.so

all: $(TARGET_LIB)

%_kernels.cu.o: distributed_embeddings/cc/kernels/%_kernels.cu distributed_embeddings/cc/kernels/%.h
	$(NVCC) -c -o $@ $< -Ithird_party/thrust/dependencies/cub -Ithird_party/thrust -Ithird_party/cuCollections/include $(CFLAGS) -I. -DGOOGLE_CUDA=1 $(CUDA_GENCODE) -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

%_kernels.cc.o: distributed_embeddings/cc/kernels/%_kernels.cc distributed_embeddings/cc/kernels/%.h
	$(CXX) -c -o $@ $< $(CFLAGS) -Wall -fPIC -I/usr/local/cuda/include

$(TARGET_LIB): $(NVCC_OBJS) $(CXX_OBJS) distributed_embeddings/cc/ops/embedding_lookup_ops.cc
	$(CXX) $(CFLAGS) -fPIC -o $@ $^ $(LDFLAGS) -L/usr/local/cuda/lib64

pip_pkg: $(TARGET_LIB)
	bash build_pip_pkg.sh

test:
	$(PYTHON_BIN_PATH) distributed_embeddings/python/ops/embedding_lookup_ops_test.py

clean:
	rm -f $(TARGET_LIB) $(NVCC_OBJS) $(CXX_OBJS)

.PHONY: all test clean
