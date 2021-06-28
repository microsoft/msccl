/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, argBuffs_t argbuffs, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, argBuffs_t argbuffs, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  struct ncclInfo info = { ncclFuncAllToAll, "AllToAll",
    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS };
  info.argbuffs = argbuffs;
  return ncclEnqueueCheck(&info);
}
