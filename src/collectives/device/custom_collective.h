/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "primitives.h"
#include "msccl_interpreter.h"

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
	  if (threadIdx.x == 0) printf("yo1\n");
    using Proto = ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>;
    runInterpreter<T, RedOp, Proto>(args, 1);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
	  if (threadIdx.x == 0) printf("yo2\n");
    runInterpreter<T, RedOp, ProtoLL128>(args, 1);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
	  if (threadIdx.x == 0) printf("yo3\n");
    runInterpreter<T, RedOp, ProtoLL>(args, 1);
  }
};
