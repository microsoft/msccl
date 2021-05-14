/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

#define SCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCCL_MAX_ITER*SCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * SCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

template<typename T, typename PRIMS_WRAPPER>
class scclFunction {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      struct ncclDevComm* comm = args->comm;
      struct scclAlgorithm* scclAlgo = &comm->scclAlgo;
      const int tid = threadIdx.x;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for SCCL cross thread synchronization
      const int bid = blockIdx.x;
      const int scclNBlocks = scclAlgo->nBlocks;
      const int rscclbid = bid % scclNBlocks; // bid within a SCCL algo
      const int scclIndex = bid / scclNBlocks; // which instance of SCCL algo
      const int nscclInstnaces = gridDim.x / scclAlgo->nBlocks; // number of SCCL aglos
      struct scclThreadBlock* scclTB = &scclAlgo->scclTB[rscclbid];
      const int channelId = scclIndex * scclAlgo->nChannels + scclTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      T * thisOutput = (T*)args->recvbuff;
      T * thisScratch = (T*)args->scratchbuff;
      int recvPeer = scclTB->recvpeer;
      int sendPeer = scclTB->sendpeer;

      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, thisOutput, channel};

      const int nranks = comm->nRanks;
      const ssize_t loopSize = (ssize_t)prims.chunkSize*nscclInstnaces;
      const ssize_t size = args->coll.count;
      const ssize_t sizePerscclChunk = (size*nranks)/scclAlgo->nchunksPerLoop;
      // SCCL flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclFlags;

      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerscclChunk; gridOffset += loopSize, iter++) {
        size_t chunkOffset = prims.initIter(sizePerscclChunk, gridOffset, nscclInstnaces, scclIndex);
        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        for (int i = 0; i < scclTB->nsteps; i++){
          struct scclTransfer* sccltran = &scclTB->transfers[i];
          // if (sccltran->type == SCCL_NO_OP) continue;
          // first wait if there is a dependence
          int8_t dependentBid = sccltran->dependentBid + scclIndex * scclNBlocks;
          int8_t dependentStep = sccltran->dependentStep;
          if (sccltran->dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scclFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }
          srcPointer = (sccltran->srcbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->srcbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          srcoffset = chunkOffset + (ssize_t) sccltran->srcoffset * sizePerscclChunk;
          dstPointer = (sccltran->dstbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->dstbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstoffset = chunkOffset + (ssize_t) sccltran->dstoffset * sizePerscclChunk;
          switch (sccltran->type) {
            case SCCL_SEND:
              prims.send(srcPointer + srcoffset, dstoffset);
              break;
            case SCCL_RECV:
              prims.recv(dstPointer + dstoffset, dstoffset);
              break;
            case SCCL_RECV_COPY_SEND:
              prims.recvCopySend(dstPointer + dstoffset, dstoffset);
              break;
            case SCCL_RECV_REDUCE_SEND:
              prims.recvReduceSend(srcPointer + srcoffset);
              break;
            case SCCL_RECV_REDUCE_COPY:
              prims.recvReduceCopy(srcPointer + srcoffset, dstPointer + dstoffset);
              break;
            case SCCL_NO_OP:
              break;
            default:
              return;
          }
          if (tid == sync_tid && sccltran->has_dependence){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
            scclFlags[bid].flag = curFlag;
          }
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper {
  const int nthreads;
  const int stepSize;
  const int chunkSize;
  ncclPrimitives<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  int nelem;

  __device__ SimpleWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      chunkSize(stepSize * SCCL_CHUNKSTEPS),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {}

  __device__ size_t initIter(ssize_t sizePerscclChunk, ssize_t gridOffset, int nscclInstnaces, int scclIndex) {
    int realChunkSize = min(chunkSize, DIVUP(sizePerscclChunk-gridOffset,nscclInstnaces));
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset + scclIndex*realChunkSize;
    nelem = min(realChunkSize, sizePerscclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset) {
    prims.directSend(chunkPointer, dstoffset, nelem);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset) {
    prims.directRecv(chunkPointer, dstoffset, nelem);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset) {
    prims.directRecvCopySend(chunkPointer, dstoffset, nelem);
  }
  
  __device__ void recvReduceSend(T * chunkPointer) {
    prims.recvReduceSend(chunkPointer, nelem);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionSimple : public scclFunction<T, SimpleWrapper<FUNC, T, UNROLL>> {};

#include "prims_ll128.h"
template<class FUNC, typename T>
struct LL128Wrapper {
  const int stepSize;
  ssize_t chunkSize;
  const ssize_t minChunkSize;
  ncclLL128Primitives<T, FUNC, 1, 1> prims;

  int nelem;

  __device__ LL128Wrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepSize(args->comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS)),
      chunkSize(stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T))),
      minChunkSize((NCCL_LL128_SHMEM_ELEMS_PER_THREAD*args->nThreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2),
      prims(tid, args->nThreads, recvPeer, sendPeer, stepSize, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerscclChunk, ssize_t gridOffset, int nscclInstnaces, int scclIndex) {
    chunkSize = min(chunkSize, DIVUP(sizePerscclChunk-gridOffset,nscclInstnaces*minChunkSize)*minChunkSize);
    ssize_t chunkOffset = gridOffset + scclIndex*chunkSize;
    nelem = min(chunkSize, sizePerscclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset) {
    prims.send(chunkPointer, nelem);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset) {
    prims.recv(chunkPointer, nelem);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset) {
    prims.recvCopySend(chunkPointer, nelem);
  }

  __device__ void recvReduceSend(T * chunkPointer) {
    prims.recvReduceSend(chunkPointer, nelem);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem);
  }  
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL128 : public scclFunction<T, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T>
struct LLWrapper {
  const int stepLines;
  const ssize_t chunkSize;
  ncclLLPrimitives<T, FUNC, 1, 1> prims;

  int nelem;

  __device__ LLWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepLines(args->comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS)),
      chunkSize(stepLines * sizeof(uint64_t) / sizeof(T)),
      prims(tid, args->nThreads, recvPeer, sendPeer, stepLines, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerscclChunk, ssize_t gridOffset, int nscclInstnaces, int scclIndex) {
    ssize_t chunkOffset = gridOffset + scclIndex*chunkSize;
    nelem = min(chunkSize, sizePerscclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset) {
    prims.send(chunkPointer, nelem);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset) {
    prims.recv(chunkPointer, nelem);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset) {
    prims.recvCopySend(chunkPointer, nelem);
  }

  __device__ void recvReduceSend(T * chunkPointer) {
    prims.recvReduceSend(chunkPointer, nelem);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem);
  }  
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL : public scclFunction<T, LLWrapper<FUNC, T>> {};

