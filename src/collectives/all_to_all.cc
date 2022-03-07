/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"
#include "collectives.h"
#include "devcomm.h"

ncclResult_t msccl2DAllToAll(const void *sendbuff, void *recvbuff, size_t sendcount,
                          ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream){
  int nGpus = comm->localRanks, nNodes = comm->nNodes;
  if (nGpus == 1 || nNodes == 1){
    WARN("number of local GPUs (%d) or number of nodes (%d) is 1.", nGpus, nNodes);
    return ncclInvalidUsage;    
  }
  // 2D Hierarchical AlltoAll algorithm
  // phase 0. per-gpu (nGpus) stride copy
  CUDACHECK(strideMemcpyAsync(recvbuff, sendbuff, sendcount * ncclTypeSize(datatype), nGpus, nNodes, stream));
  // phase 1. intra-node alltoall
  NCCLCHECK(ncclGroupStart());
  for (int g = 0; g < nGpus; g++)
  {
    NCCLCHECK(ncclSend(((char *)recvbuff) + g * nNodes * sendcount * ncclTypeSize(datatype), nNodes * sendcount, datatype, g + comm->node * nGpus, comm, stream));
    NCCLCHECK(ncclRecv(((char *)sendbuff) + g * nNodes * sendcount * ncclTypeSize(datatype), nNodes * sendcount, datatype, g + comm->node * nGpus, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  // phase 2. per-gpu (nNodes) stride copy
  CUDACHECK(strideMemcpyAsync(recvbuff, sendbuff, sendcount * ncclTypeSize(datatype), nNodes, nGpus, stream));
  // phase 3. inter-node alltoall
  NCCLCHECK(ncclGroupStart());
  for (int n = 0; n < nNodes; n++)
  {
    NCCLCHECK(ncclSend(((char *)recvbuff) + n * nGpus * sendcount * ncclTypeSize(datatype), nGpus * sendcount, datatype, n * nGpus + comm->cudaDev, comm, stream));
    NCCLCHECK(ncclRecv(((char *)sendbuff) + n * nGpus * sendcount * ncclTypeSize(datatype), nGpus * sendcount, datatype, n * nGpus + comm->cudaDev, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  CUDACHECK(cudaMemcpyAsync(recvbuff, sendbuff, comm->nRanks * sendcount * ncclTypeSize(datatype), cudaMemcpyDeviceToDevice, stream));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclAllToAll, const void *sendbuff, void *recvbuff, size_t sendcount,
         ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclAllToAll(const void *sendbuff, void *recvbuff, size_t sendcount,
                          ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream)
{
  if (sendcount == 0) return ncclSuccess;

  size_t allcount = sendcount * comm->nRanks;
  size_t nbytes = allcount * ncclTypeSize(datatype);
  int nGpus = comm->localRanks, nNodes = comm->nNodes;

  if (comm->nScclRegistrations > 0)
  {
    for (int i = 0; i < comm->nScclRegistrations; ++i)
    {
      struct scclRegistration *reg = &comm->scclRegistrations[i];
      if (reg->minBytes <= nbytes && (nbytes < reg->maxBytes || reg->maxBytes == -1))
      {
        struct scclAlgorithm *scclAlgo = &comm->scclAlgos[reg->algoIndex];
        if ((scclAlgo->isValid) && (scclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == scclAlgo->ngpus) && ((allcount % scclAlgo->nchunksPerLoop) == 0))
        {
          if !strcmp(scclAlgo->name, "2D") {
            return msccl2DAllToAll(sendbuff, recvbuff, sendcount, datatype, comm, stream);
          } else {
            NVTX3_FUNC_RANGE_IN(nccl_domain);
            struct ncclInfo info = {ncclFuncAllToAll, "AllToAll",
                                    sendbuff, recvbuff, 0 /* all-to-all can only be out of place */, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
                                    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS};
            info.scclAlgoIndex = reg->algoIndex;
            auto curProto = scclAlgo->protocol;
            scclAlgo->protocol = reg->protocol;
            auto ret = ncclEnqueueCheck(&info);
            scclAlgo->protocol = curProto;
            return ret;
          }
        }
      }
    }
  }
  else
  {
    for (int scclAlgoIndex = 0; scclAlgoIndex < comm->numberOfSCCAlgorithms; scclAlgoIndex++)
    {
      struct scclAlgorithm *scclAlgo = &comm->scclAlgos[scclAlgoIndex];
      if ((scclAlgo->isValid) && (scclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == scclAlgo->ngpus) && ((allcount % comm->scclAlgos[scclAlgoIndex].nchunksPerLoop) == 0) && (nbytes >= scclAlgo->minBytes) && (nbytes < scclAlgo->maxBytes))
      {
        if !strcmp(scclAlgo->name, "2D") {
          return msccl2DAllToAll(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        } else {
          NVTX3_FUNC_RANGE_IN(nccl_domain);
          struct ncclInfo info = {ncclFuncAllToAll, "AllToAll",
                                  sendbuff, recvbuff, 0 /* all-to-all can only be out of place */, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
                                  SCCL_CHUNKSTEPS, SCCL_SLICESTEPS};
          info.scclAlgoIndex = scclAlgoIndex;
          return ncclEnqueueCheck(&info);
        }
      }
    }
  }

  // default p2p
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++)
  {
    NCCLCHECK(ncclSend(((char *)sendbuff) + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, r, comm, stream));
    NCCLCHECK(ncclRecv(((char *)recvbuff) + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
