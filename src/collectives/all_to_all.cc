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
  if (sendbuff == recvbuff){
    WARN("In-place alltoall is not possible currently.");
    return ncclInvalidUsage;
  }

  size_t allcount = sendcount * comm->nRanks;
  size_t nbytes = allcount * ncclTypeSize(datatype);

  struct mscclHostCommInfo* mscclInfo = &comm->mscclHostComm;
  if (mscclInfo->nMscclRegistrations > 0)
  {
    for (int i = 0; i < mscclInfo->nMscclRegistrations; ++i)
    {
      struct mscclRegistration *reg = &mscclInfo->mscclRegistrations[i];
      if (reg->minBytes <= nbytes && (nbytes < reg->maxBytes || reg->maxBytes == -1))
      {
        struct mscclAlgorithm *mscclAlgo = &mscclInfo->mscclDevComm.mscclAlgos[reg->algoIndex];
        if ((mscclAlgo->isValid) && (mscclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == mscclAlgo->ngpus) && ((allcount % mscclAlgo->nchunksPerLoop) == 0))
        {
          // if it was the 2D algorithm, select it first.
          if (!strcmp(mscclAlgo->name, "2D")) {
            return msccl2DAllToAll(sendbuff, recvbuff, sendcount, datatype, comm, stream);
          } else {
            NVTX3_FUNC_RANGE_IN(nccl_domain);
            struct ncclInfo info = {ncclFuncAllToAll, "AllToAll",
                                    sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
                                    MSCCL_CHUNKSTEPS, MSCCL_SLICESTEPS};
            info.algorithm = NCCL_ALGO_MSCCL;
            info.mscclInfo.mscclAlgoIndex = reg->algoIndex;
            info.protocol = reg->protocol;
            return ncclEnqueueCheck(&info);
          }
        }
      }
    }
  }
  else
  {
    for (int mscclAlgoIndex = 0; mscclAlgoIndex < mscclInfo->numberOfMSCCLAlgorithms; mscclAlgoIndex++)
    {
      struct mscclAlgorithm *mscclAlgo = &mscclInfo->mscclDevComm.mscclAlgos[mscclAlgoIndex];
      if ((mscclAlgo->isValid) && (mscclAlgo->collectiveType == ncclFuncAllToAll) && (comm->nRanks == mscclAlgo->ngpus) 
        && ((allcount % mscclAlgo->nchunksPerLoop) == 0) && (nbytes >= mscclAlgo->minBytes) && (nbytes < mscclAlgo->maxBytes))
      {
        // if it was the 2D algorithm, select it first.
        if (!strcmp(mscclAlgo->name, "2D")) {
          return msccl2DAllToAll(sendbuff, recvbuff, sendcount, datatype, comm, stream);
        } else {
          NVTX3_FUNC_RANGE_IN(nccl_domain);
          struct ncclInfo info = {ncclFuncAllToAll, "AllToAll",
                                  sendbuff, recvbuff, sendcount, datatype, ncclSum, 0, comm, stream, /* Args */
                                  MSCCL_CHUNKSTEPS, MSCCL_SLICESTEPS};
          info.algorithm = NCCL_ALGO_MSCCL;
          info.mscclInfo.mscclAlgoIndex = mscclAlgoIndex;
          info.protocol = mscclAlgo->protocol;
          return ncclEnqueueCheck(&info);
        }
      }
    }
  }

  // default p2p if all failed
  NCCLCHECK(ncclGroupStart());
  for (int r = 0; r < comm->nRanks; r++)
  {
    NCCLCHECK(ncclSend(((char *)sendbuff) + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, r, comm, stream));
    NCCLCHECK(ncclRecv(((char *)recvbuff) + r * sendcount * ncclTypeSize(datatype), sendcount, datatype, r, comm, stream));
  }
  NCCLCHECK(ncclGroupEnd());
  return ncclSuccess;
}
