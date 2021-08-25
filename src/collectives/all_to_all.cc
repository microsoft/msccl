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
  
  if (comm->scclAlgo.isValid && ((sendcount*comm->nRanks) % comm->scclAlgo.nchunksPerLoop) == 0){
    NVTX3_FUNC_RANGE_IN(nccl_domain);
    struct ncclInfo info = { ncclFuncAllToAll, "AllToAll",
      sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
      SCCL_CHUNKSTEPS, SCCL_SLICESTEPS };
    return ncclEnqueueCheck(&info);
  } else {
    NCCLCHECK(ncclGroupStart());
    for (int r=0; r<comm->nRanks; r++){
      if (sendcount != 0){
        NCCLCHECK(ncclSend(((char*)sendbuff)+r*sendcount*ncclTypeSize(datatype), sendcount, datatype, r, comm, stream));
        NCCLCHECK(ncclRecv(((char*)recvbuff)+r*sendcount*ncclTypeSize(datatype), sendcount, datatype, r, comm, stream));
      }
    }
    NCCLCHECK(ncclGroupEnd());   
    return ncclSuccess;
  }
}
