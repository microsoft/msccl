/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

#define SCKL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCKL_MAX_ITER*SCKL_MAX_NUM_STEPS*__WORKINDEX__ + (__GRIDOFFSET_ITER__ * SCKL_MAX_NUM_STEPS + __STEP__)

template<class FUNC, typename T, int UNROLL>
class SCKLFunctionSimple {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      struct ncclDevComm* comm = args->comm;
      struct scklAlgorithm* scklAlgo = &comm->scklAlgo;
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads-WARP_SIZE;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for sckl cross thread synchronization
      const int bid = blockIdx.x;
      const int scklNBlocks = scklAlgo->nBlocks;
      const int rscklbid = bid % scklNBlocks; // bid within a sckl algo
      const int scklIndex = bid / scklNBlocks; // which instance of sckl algo
      const int nScklInstnaces = gridDim.x / scklAlgo->nBlocks; // number of sckl aglos
      struct scklThreadBlock* scklTB = &scklAlgo->scklTB[rscklbid];
      const int channelId = scklIndex * scklAlgo->nChannels + scklTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;
      const int stepSize = comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS);
      const int chunkSize = stepSize * SCKL_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const int nchunksPerLoopPerRank = scklAlgo->nchunksPerLoop/nranks;
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
      int recvPeer = scklTB->recvpeer;
      int sendPeer = scklTB->sendpeer;

      ncclPrimitives<UNROLL, SCKL_CHUNKSTEPS/SCKL_SLICESTEPS, SCKL_SLICESTEPS, T, 1, 1, 1, FUNC>
        prims(tid, nthreads, &recvPeer, &sendPeer, thisOutput, stepSize, channel, comm, ncclShmem->ptrs, 0);
      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerScklChunk; gridOffset += loopSize, iter++) {
        int realChunkSize = min(chunkSize, DIVUP(sizePerScklChunk-gridOffset,nScklInstnaces));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + scklIndex*realChunkSize;
        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        int nelem = min(realChunkSize, sizePerScklChunk-chunkOffset);
        for (int i = 0; i < scklTB->nsteps; i++){
          struct scklTransfer* sckltran = &scklTB->transfers[i];
          if (sckltran->type == SCKL_NO_OP) continue;
          // first wait if there is a dependence
          int8_t dependentBid = sckltran->dependentBid + scklIndex * scklNBlocks;
          int8_t dependentStep = sckltran->dependentStep;
          if (sckltran->dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scklFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }

          srcPointer = (sckltran->srcbuffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          srcoffset = chunkOffset + (ssize_t) sckltran->srcoffset * sizePerScklChunk;
          dstPointer = (sckltran->dstbuffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          dstoffset = chunkOffset + (ssize_t) sckltran->dstoffset * sizePerScklChunk;
          switch (sckltran->type) {
            case SCKL_SEND:
              prims.directSend(srcPointer + srcoffset, dstoffset, nelem);
              break;
            case SCKL_RECV:
              prims.directRecv(dstPointer + dstoffset, dstoffset, nelem);
              break;
            case SCKL_RECV_COPY_SEND:
              prims.directRecvCopySend(dstPointer + dstoffset, dstoffset, nelem);
              break;
            default:
              return;
          }

          if (tid == sync_tid){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
            scklFlags[bid].flag = curFlag;
          }
        }
      }
    }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class SCKLFunctionLL128 {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      struct ncclDevComm* comm = args->comm;
      struct scklAlgorithm* scklAlgo = &comm->scklAlgo;
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for sckl cross thread synchronization
      const int bid = blockIdx.x;
      const int scklNBlocks = scklAlgo->nBlocks;
      const int rscklbid = bid % scklNBlocks; // bid within a sckl algo
      const int scklIndex = bid / scklNBlocks; // which instance of sckl algo
      const int nScklInstnaces = gridDim.x / scklAlgo->nBlocks; // number of sckl aglos
      struct scklThreadBlock* scklTB = &scklAlgo->scklTB[rscklbid];
      const int channelId = scklIndex * scklAlgo->nChannels + scklTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;
      const int stepSize = comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS);
      ssize_t chunkSize = stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T));
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2;
      const int nranks = comm->nRanks;
      const int nchunksPerLoopPerRank = scklAlgo->nchunksPerLoop/nranks;
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
      int recvPeer = scklTB->recvpeer;
      int sendPeer = scklTB->sendpeer;

      ncclLL128Primitives<T, FUNC, 1, 1> prims(tid, nthreads, &recvPeer, &sendPeer, stepSize, channel, comm);
      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerScklChunk; gridOffset += loopSize, iter++) {
        chunkSize = min(chunkSize, DIVUP(sizePerScklChunk-gridOffset,nScklInstnaces*minChunkSize)*minChunkSize);
        ssize_t chunkOffset = gridOffset + scklIndex*chunkSize;
        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        int nelem = min(chunkSize, sizePerScklChunk-chunkOffset);
        for (int i = 0; i < scklTB->nsteps; i++){
          struct scklTransfer* sckltran = &scklTB->transfers[i];
          if (sckltran->type == SCKL_NO_OP) continue;
          // first wait if there is a dependence
          int8_t dependentBid = sckltran->dependentBid + scklIndex * scklNBlocks;
          int8_t dependentStep = sckltran->dependentStep;
          if (sckltran->dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scklFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }

          srcPointer = (sckltran->srcbuffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          srcoffset = chunkOffset + (ssize_t) sckltran->srcoffset * sizePerScklChunk;
          dstPointer = (sckltran->dstbuffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          dstoffset = chunkOffset + (ssize_t) sckltran->dstoffset * sizePerScklChunk;
          switch (sckltran->type) {
            case SCKL_SEND:
              prims.send(srcPointer + srcoffset, nelem);
              break;
            case SCKL_RECV:
              prims.recv(dstPointer + dstoffset, nelem);
              break;
            case SCKL_RECV_COPY_SEND:
              prims.recvCopySend(dstPointer + dstoffset, nelem);
              break;
            default:
              return;
          }
          if (tid == sync_tid){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
            scklFlags[bid].flag = curFlag;
          }
        }
      }
    }
};


template<class FUNC, typename T, int UNROLL>
class SCKLFunctionLL {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      struct ncclDevComm* comm = args->comm;
      struct scklAlgorithm* scklAlgo = &comm->scklAlgo;
      const int tid = threadIdx.x;
      const int nthreads = args->nThreads;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for sckl cross thread synchronization
      const int bid = blockIdx.x;
      const int scklNBlocks = scklAlgo->nBlocks;
      const int rscklbid = bid % scklNBlocks; // bid within a sckl algo
      const int scklIndex = bid / scklNBlocks; // which instance of sckl algo
      const int nScklInstnaces = gridDim.x / scklAlgo->nBlocks; // number of sckl aglos
      struct scklThreadBlock* scklTB = &scklAlgo->scklTB[rscklbid];
      const int channelId = scklIndex * scklAlgo->nChannels + scklTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;
      const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const int nranks = comm->nRanks;
      const int nchunksPerLoopPerRank = scklAlgo->nchunksPerLoop/nranks;
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
      int recvPeer = scklTB->recvpeer;
      int sendPeer = scklTB->sendpeer;

      ncclLLPrimitives<T, FUNC, 1, 1> prims(tid, nthreads, &recvPeer, &sendPeer, stepLines, channel, comm);
      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerScklChunk; gridOffset += loopSize, iter++) {
        ssize_t chunkOffset = gridOffset + scklIndex*chunkSize;
        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        int nelem = min(chunkSize, sizePerScklChunk-chunkOffset);
        for (int i = 0; i < scklTB->nsteps; i++){
          struct scklTransfer* sckltran = &scklTB->transfers[i];
          if (sckltran->type == SCKL_NO_OP) continue;
          // first wait if there is a dependence
          int8_t dependentBid = sckltran->dependentBid + scklIndex * scklNBlocks;
          int8_t dependentStep = sckltran->dependentStep;
          if (sckltran->dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scklFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }

          srcPointer = (sckltran->srcbuffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          srcoffset = chunkOffset + (ssize_t) sckltran->srcoffset * sizePerScklChunk;
          dstPointer = (sckltran->dstbuffer == SCKL_INPUT_BUFFER) ? thisInput : thisOutput;
          dstoffset = chunkOffset + (ssize_t) sckltran->dstoffset * sizePerScklChunk;
          switch (sckltran->type) {
            case SCKL_SEND:
              prims.send(srcPointer + srcoffset, nelem);
              break;
            case SCKL_RECV:
              prims.recv(dstPointer + dstoffset, nelem);
              break;
            case SCKL_RECV_COPY_SEND:
              prims.recvCopySend(dstPointer + dstoffset, nelem);
              break;
            default:
              return;
          }
          if (tid == sync_tid && sckltran->has_dependence){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
            scklFlags[bid].flag = curFlag;
          }
        }
      }
    }
};