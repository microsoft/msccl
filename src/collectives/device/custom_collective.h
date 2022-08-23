/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "primitives.h"
#include "msccl_interpreter.h"

template<typename T, typename RedOp, int Proto>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, Proto> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    RunWorkElement<ncclFuncAllReduce, T, RedOp, NCCL_ALGO_MSCCL, Proto>().run(args);
  }
};