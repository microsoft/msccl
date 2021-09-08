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
  size_t allcount = sendcount*comm->nRanks;
  size_t nbytes = allcount*ncclTypeSize(datatype);
  for (int scclAlgoIndex = 0; scclAlgoIndex < comm->numberOfSCCAlgorithms; scclAlgoIndex++) {
    struct scclAlgorithm* scclAlgo = &comm->scclAlgos[scclAlgoIndex];
    if ((scclAlgo->isValid) && (scclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == scclAlgo->ngpus) 
        && ((allcount % comm->scclAlgos[scclAlgoIndex].nchunksPerLoop) == 0)
        && (nbytes >= scclAlgo->minBytes) && (nbytes < scclAlgo->maxBytes)){
      NVTX3_FUNC_RANGE_IN(nccl_domain);
      struct ncclInfo info = { ncclFuncAllToAll, "AllToAll",
        sendbuff, recvbuff, 0 /* all-to-all can only be out of place */, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
        SCCL_CHUNKSTEPS, SCCL_SLICESTEPS };
      info.scclAlgoIndex = scclAlgoIndex;
      return ncclEnqueueCheck(&info);
    }
  }
  // If there is no proper SCCL algorithm, then use p2p
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
