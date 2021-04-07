/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(realChunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          prims.directSend(thisInput+chunkOffset, offset, nelem);
        } else {
          prims.directCopySend(thisInput+chunkOffset, thisOutput+offset, offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          prims.directRecvCopySend(thisOutput+offset, offset, nelem);
        }

        // Make final copy from buffer to dest.
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        // Final wait/copy.
        prims.directRecv(thisOutput+offset, offset, nelem);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          LLprims.send(thisInput+chunkOffset, nelem);
        } else {
          LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(thisOutput+offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(thisOutput+offset, nelem);
      }
    }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int bid = args->coll.bid;
      const int nChannels = args->coll.nChannels;
      struct ncclDevComm* comm = args->comm;
      struct ncclChannel* channel = comm->channels+blockIdx.x;
      struct ncclRing* ring = &channel->ring;
      const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;

      ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);

        ssize_t chunkOffset = gridOffset + bid*chunkSize;

        /////////////// begin AllGather steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        if (thisInput + chunkOffset == thisOutput + offset) { // In place
          LLprims.send(thisInput+chunkOffset, nelem);
        } else {
          LLprims.copySend(thisInput+chunkOffset, thisOutput+offset, nelem);
        }

        // k-2 steps: copy to next GPU
        for (int j=1; j<nranks-1; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvCopySend(thisOutput+offset, nelem);
        }

        // step k-1: final store
        rankDest = ring->devUserRanks[1];
        offset = chunkOffset + rankDest * size;

        LLprims.recv(thisOutput+offset, nelem);
      }
    }
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_TREE, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_COLLNET, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

#define SCKL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCKL_MAX_ITER*SCKL_MAX_NUM_STEPS*__WORKINDEX__ + (__GRIDOFFSET_ITER__ * SCKL_MAX_NUM_STEPS + __STEP__)


template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllGather, NCCL_ALGO_SCKL, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      struct ncclDevComm* comm = args->comm;
      struct scklAlgorithm* scklAlgo = &comm->scklAlgo;
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int sync_tid = 0; //args->nThreads-1;
      const int bid = blockIdx.x;
      const int nChannels = args->coll.nChannels;
      const int scklNBlocks = scklAlgo->nBlocks;
      const int rscklbid = bid % scklNBlocks; // bid within a sckl algo
      const int scklIndex = bid / scklNBlocks; // which instance of sckl algo
      const int nScklInstnaces = gridDim.x / scklAlgo->nBlocks; // number of sckl aglos
      struct scklThreadBlock* scklTB = &scklAlgo->scklTB[rscklbid];
      const int channelId = scklIndex * scklAlgo->nChannels + scklTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * ALLTOALL_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const int nchunksPerLoopPerRank = scklAlgo->nchunksPerLoop/nranks;
      const int totalNChunksPerLoopPerRank = nScklInstnaces*nchunksPerLoopPerRank;
      const ssize_t loopSize = (ssize_t)chunkSize*nScklInstnaces;
      const ssize_t size = args->coll.count;
      const ssize_t sizePerScklChunk = size/nchunksPerLoopPerRank;
      // sckl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scklFlag* scklFlags = comm->scklFlags;
      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      T * thisOutput = (T*)args->recvbuff;
      int myRank = channel->ring.devUserRanks[0];
      int m1 = -1;
      int recvPeer = (scklTB->type == SCKL_RECV) ? scklTB->peer : m1;
      int sendPeer = (scklTB->type == SCKL_SEND) ? scklTB->peer : m1;

      ncclPrimitives<UNROLL, ALLTOALL_CHUNKSTEPS/ALLTOALL_SLICESTEPS, ALLTOALL_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &recvPeer, &sendPeer, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerScklChunk; gridOffset += loopSize, iter++) {
        int realChunkSize = min(chunkSize, DIVUP(sizePerScklChunk-gridOffset,nScklInstnaces));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + scklIndex*realChunkSize;
        ssize_t offset;
        int nelem = min(realChunkSize, sizePerScklChunk-chunkOffset);
        for (int i = 0; i < scklTB->nsteps; i++){

          struct scklTransfer* sckltran = &scklTB->transfers[i];
          if (sckltran->offset == -1) continue;
          offset = chunkOffset + sckltran->offset * sizePerScklChunk;
          T* thisbuffer = (sckltran->buffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          if (scklTB->type == SCKL_SEND){
            int8_t dependentBid = sckltran->dependentBid + scklIndex * scklNBlocks;
            int8_t dependentStep = sckltran->dependentStep;
            if (sckltran->dependentBid >= 0){
              if (tid == sync_tid){
                uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
                while ((scklFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
            }
            prims.directSend(thisbuffer + offset, offset, nelem);
          } else if (scklTB->type == SCKL_RECV) {
            prims.directRecv(thisbuffer + offset, offset, nelem);
            if (tid == sync_tid){
              __threadfence();
              uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
              scklFlags[bid].flag = curFlag;
            }
          }
        }
      }
    }
};
