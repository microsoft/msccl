/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "msccl_interpreter.h"

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllToAll, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    using Proto = ProtoSimple<MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS>;
    runInterpreter<T, RedOp, Proto>(args, ncclShmem.comm.nRanks);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllToAll, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_LL128> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runInterpreter<T, RedOp, ProtoLL128>(args, ncclShmem.comm.nRanks);
  }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncAllToAll, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_LL> {
  __device__ __forceinline__ void run(ncclWorkElem *args) {
    runInterpreter<T, RedOp, ProtoLL>(args, ncclShmem.comm.nRanks);
  }
};
