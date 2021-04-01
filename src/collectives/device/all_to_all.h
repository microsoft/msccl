/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<int ALGO, int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllToAll, ALGO, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = blockIdx.x;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      const int scklNumBlocksPerChannel = args->scklNumBlocksPerChannel;
      const int channelId = bid/scklNumBlocksPerChannel;
      struct ncclChannel* channel = comm->channels+channelId;
      // relative bid to a channel
      int rbid = bid % scklNumBlocksPerChannel;
      struct scklAlgorithm* scklAlgo = &comm->scklAlgo;
      struct scklThreadBlock* sckltb = &scklAlgo->scklTB[rbid];
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * ALLTOALL_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;
      const int nChunks = scklAlgo->nChunks;
      // assume that size is divisible by nchunks
      const ssize_t sizePerChunk = size/nChunks;
      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;
      int myRank = channel->ring.devUserRanks[0];
      int m1 = -1;
      int recvPeer = (sckltb->type == SCKL_RECV) ? sckltb->peer : m1;
      int sendPeer = (sckltb->type == SCKL_SEND) ? sckltb->peer : m1;

      ncclPrimitives<UNROLL, ALLTOALL_CHUNKSTEPS/ALLTOALL_SLICESTEPS, ALLTOALL_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &recvPeer, &sendPeer, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

      for (ssize_t gridOffset = 0; gridOffset < sizePerChunk; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(sizePerChunk-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + channelId*realChunkSize;
        ssize_t offset;
        int nelem = min(realChunkSize, sizePerChunk-chunkOffset);
        for (int i = 0; i < sckltb->nsteps; i++){
          offset = chunkOffset + sckltb->transfers[i] * sizePerChunk;
          if (sckltb->type == SCKL_SEND){
            prims.directSend(thisInput + offset, offset, nelem);
          } else if (sckltb->type == SCKL_RECV) {
            prims.directRecv(thisOutput + offset, offset, nelem);
          }
        }
      }
    }
};
