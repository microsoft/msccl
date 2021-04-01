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

#define GLOBALITERSTEP(GLOBALITER,STEP) \
  (GLOBALITER * SCKL_MAX_NUM_STEPS + STEP)

#define SETFLAG(GLOBALITER,STEP,WORKINDEX,FLAG) \
  FLAG = NCCL_MAX_OPS * GLOBALITERSTEP(GLOBALITER,STEP) + WORKINDEX
#define GETFLAGITEMS(GLOBALITERSTEP,WORKINDEX,FLAG) \
  WORKINDEX = FLAG % NCCL_MAX_OPS; \
  GLOBALITERSTEP = FLAG / NCCL_MAX_OPS

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
      const int workIndex = args->index+1; // sckl flags all start out with 0. 
      volatile uint64_t* scklFlags = comm->scklFlags;
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
          T* thisbuffer = (sckltran->buffer == SCKL_THIS_INPUT) ? thisInput : thisOutput;
          if (sckltb->type == SCKL_SEND){
            int8_t dependence = sckltran->dependence;
            int8_t dependenceStep = sckltran->dependenceStep;
            if (dependence >= 0){
              if (tid == 0){
                uint64_t readFlag;
                int readGlobalIterStep;
                int readWorkIndex;
                int gaolGlobalIterStep = GLOBALITERSTEP(iter, dependenceStep);
                do {
                  readFlag = *(scklFlags + dependence);
                  GETFLAGITEMS(readGlobalIterStep, readWorkIndex, readFlag);
                } while (readWorkIndex != workIndex || readGlobalIterStep < gaolGlobalIterStep);
              }
              __syncthreads();
            }
            prims.directSend(thisbuffer + offset, offset, nelem);
          } else if (sckltb->type == SCKL_RECV) {
            prims.directRecv(thisbuffer + offset, offset, nelem);
            if (tid == 0){
              uint64_t curFlag;
              SETFLAG(iter, i, workIndex, curFlag);
              scklFlags[bid] = curFlag;
            }
          }
        }
      }
    }
};
