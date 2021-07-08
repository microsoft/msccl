/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"

NCCL_API(ncclResult_t, ncclAllToAll, const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllToAll(const void* sendbuff, void* recvbuff, size_t sendcount,
    ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  
  if (comm->scclAlgo.isValid && ((sendcount*comm->nRanks) % comm->scclAlgo.nchunksPerLoop) == 0){
    struct ncclInfo info = { ncclFuncAllToAll, "AllToAll",
      sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
      SCCL_CHUNKSTEPS, SCCL_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<comm->nRanks; r++){
      struct ncclInfo info = { ncclFuncSendRecv, "Send",
        ((char*)sendbuff)+r*sendcount*ncclTypeSize(datatype), NULL, sendcount, datatype, ncclSum, r, comm, stream, /* Args */
        1, 1 };
      ncclResult_t ret;
      ret = ncclEnqueueCheck(&info);
      if (ret != ncclSuccess)
        return ret;
    }
    for (int r=0; r<comm->nRanks; r++){
      struct ncclInfo info = { ncclFuncSendRecv, "Recv",
        NULL, ((char*)recvbuff)+r*sendcount*ncclTypeSize(datatype), sendcount, datatype, ncclSum, r, comm, stream, /* Args */
        1, 1 };
      ncclResult_t ret;
      ret = ncclEnqueueCheck(&info);
      if (ret != ncclSuccess)
        return ret;
    }
    NCCLCHECK(ncclGroupEnd());   
    return ncclSuccess;
  }
}
