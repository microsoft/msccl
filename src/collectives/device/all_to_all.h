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
  private:

#define SCKL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCKL_MAX_ITER*SCKL_MAX_NUM_STEPS*__WORKINDEX__ + (__GRIDOFFSET_ITER__ * SCKL_MAX_NUM_STEPS + __STEP__)

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
      // sckl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scklFlag* scklFlags = comm->scklFlags;
      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      T * thisOutput = (T*)args->recvbuff;
      int myRank = channel->ring.devUserRanks[0];
      int m1 = -1;
      int recvPeer = (sckltb->type == SCKL_RECV) ? sckltb->peer : m1;
      int sendPeer = (sckltb->type == SCKL_SEND) ? sckltb->peer : m1;

      ncclPrimitives<UNROLL, ALLTOALL_CHUNKSTEPS/ALLTOALL_SLICESTEPS, ALLTOALL_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &recvPeer, &sendPeer, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerChunk; gridOffset += loopSize, iter++) {
        int realChunkSize = min(chunkSize, DIVUP(sizePerChunk-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + channelId*realChunkSize;
        ssize_t offset;
        int nelem = min(realChunkSize, sizePerChunk-chunkOffset);
        for (int i = 0; i < sckltb->nsteps; i++){
          struct scklTransfer* sckltran = &sckltb->transfers[i];
          if (sckltran->offset == -1) continue;
          offset = chunkOffset + sckltran->offset * sizePerChunk;
          T* thisbuffer = (sckltran->buffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          if (sckltb->type == SCKL_SEND){
            int8_t dependentBid = sckltran->dependentRbid + scklNumBlocksPerChannel * channelId;
            int8_t dependentStep = sckltran->dependentStep;
            if (sckltran->dependentRbid >= 0){
              if (tid == 0){
                uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
                while ((scklFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
            }
            prims.directSend(thisbuffer + offset, offset, nelem);
          } else if (sckltb->type == SCKL_RECV) {
            prims.directRecv(thisbuffer + offset, offset, nelem);
            if (tid == 0){
              uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
              scklFlags[bid].flag = curFlag;
            }
          }
        }
      }
    }
};
