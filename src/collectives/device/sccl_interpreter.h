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

template<class FUNC, typename T, typename PRIMS_WRAPPER>
class scclFunction {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      struct scclAlgorithm* scclAlgo = &comm->scclAlgos[args->scclAlgoIndex];
      const int tid = threadIdx.x;
      const int nThreads = args->nThreads; // last thread is most likely not doing anthing and used for SCCL cross thread synchronization
      const int bid = blockIdx.x;
      struct scclThreadBlock* scclTB = &scclAlgo->scclTB[bid];
      const int channelId = scclTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      T * thisOutput = (T*)args->recvbuff;
      T * thisScratch = (T*)args->scratchbuff;
      int recvPeer = scclTB->recvpeer;
      int sendPeer = scclTB->sendpeer;

      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, thisOutput, channel};

      const ssize_t loopSize = (ssize_t)prims.chunkSize;
      const ssize_t size = args->coll.count;
      const ssize_t sizePerScclChunk = (size*sizeMultiplier)/scclAlgo->nchunksPerLoop;
      uint32_t scclMaxAllowedCount = args->scclMaxAllowedCount;

      // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclAlgoShared.flags;

      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerScclChunk; gridOffset += loopSize, iter++) {
        size_t chunkOffset = prims.initIter(sizePerScclChunk, gridOffset);
        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        int step = 0;
        for (int i = 0; i < scclTB->nsteps; i++){
          struct scclTransfer* sccltran = &scclTB->transfers[i];
          // first wait if there is a dependence
          int16_t dependentPointer = sccltran->depencePointer;
          int16_t numDependences = sccltran->numDependences;
          if (sccltran->numDependences > 0){
            for (int index = tid; index < numDependences; index += nThreads) {
              int8_t dependentBid = scclTB->dependentBid[dependentPointer+index];
              int16_t dependentStep = scclTB->dependentStep[dependentPointer+index];
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scclFlags + dependentBid)->flag < goalFlag){};
            }
            step += sccltran->numDependences-1;
            __syncthreads();
          }

          srcPointer = (sccltran->srcbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->srcbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstPointer = (sccltran->dstbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->dstbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          int count = sccltran->count;
          if (sccltran->type == SCCL_REDUCE){
            int numReductions = sccltran->numReductions;
            int thisChunkSize = prims.nelem * count;
            dstoffset = chunkOffset + (ssize_t) (sccltran->dstoffset) * sizePerScclChunk;
            for (int r = 0; r < numReductions; r++){
              srcoffset = chunkOffset + (ssize_t) (scclTB->reductionSrcOffsets[sccltran->reductionPointer+r]) * sizePerScclChunk;
              for (int index = tid; index < thisChunkSize; index += nThreads){
                T c = dstPointer[dstoffset + index];
                T t = srcPointer[srcoffset + index];
                c = FUNC()(c, t);
                dstPointer[dstoffset + index] = c;
              }
            }
            step += numReductions-1;
          } else {
            for (int c = 0; c < count; c += scclMaxAllowedCount) {
              srcoffset = chunkOffset + (ssize_t) (sccltran->srcoffset+c) * sizePerScclChunk;
              dstoffset = chunkOffset + (ssize_t) (sccltran->dstoffset+c) * sizePerScclChunk;
              int thisCount = min(scclMaxAllowedCount, count-c);
              if (sccltran->type == SCCL_SEND)
                prims.send(srcPointer + srcoffset, dstoffset, thisCount);
              else if (sccltran->type == SCCL_RECV)
                prims.recv(dstPointer + dstoffset, dstoffset, thisCount);
              else if (sccltran->type == SCCL_REDUCE){
                int numReductions = sccltran->numReductions;
                int thisChunkSize = prims.nelem * thisCount;
                for (int index = tid; index < thisChunkSize; index += nThreads){
                  T c = dstPointer[dstoffset + index];
                  for (int r = 0; r < numReductions; r++){
                    srcoffset = chunkOffset + (ssize_t) (scclTB->reductionSrcOffsets[sccltran->reductionPointer+r]) * sizePerScclChunk + index;
                    T t = srcPointer[srcoffset];
                    c = FUNC()(c, t);
                  }
                  dstPointer[dstoffset + index] = c;
                }
                step += numReductions-1;
              } else if (sccltran->type == SCCL_RECV_COPY_SEND)
                prims.recvCopySend(dstPointer + dstoffset, dstoffset, thisCount);
              else if (sccltran->type == SCCL_RECV_REDUCE_SEND)
                prims.recvReduceSend(srcPointer + srcoffset, thisCount);
              else if (sccltran->type == SCCL_RECV_REDUCE_COPY_SEND)
                prims.recvReduceCopySend(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
              else if (sccltran->type == SCCL_RECV_REDUCE_COPY)
                prims.recvReduceCopy(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
              else if (sccltran->type == SCCL_LOCAL_COPY)
                prims.localCopy(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
              else
                return;
            }
          }
          if (sccltran->has_dependence){
            __syncthreads();
            if (tid == nThreads-1){
              __threadfence();
              uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, step);
              scclFlags[bid].flag = curFlag;
            }
          }
          step++;
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper {
  const int nthreads;
  const int stepSize;
  const int chunkSize;
  int nelem;

  ncclPrimitives<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ SimpleWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
    stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
    chunkSize(stepSize * SCCL_CHUNKSTEPS),
    prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    int realChunkSize = min(chunkSize, sizePerScclChunk-gridOffset);
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset;
    nelem = min(realChunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.directSend(chunkPointer, dstoffset, nelem*count);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.directRecv(chunkPointer, dstoffset, nelem*count);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.directRecvCopySend(chunkPointer, dstoffset, nelem*count);
  }

  __device__ void recvReduceSend(T * chunkPointer, int count) {
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void recvReduceCopySend(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionSimple : public scclFunction<FUNC, T, SimpleWrapper<FUNC, T, UNROLL>> {};

#include "prims_ll128.h"
template<class FUNC, typename T>
struct LL128Wrapper {
  const int stepSize;
  ssize_t chunkSize;
  const ssize_t minChunkSize;
  int nelem;

  ncclLL128Primitives<T, FUNC, 1, 1> prims;

  __device__ LL128Wrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepSize(args->comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS)),
    chunkSize(stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T))),
    minChunkSize((NCCL_LL128_SHMEM_ELEMS_PER_THREAD*args->nThreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2),
    prims(tid, args->nThreads, recvPeer, sendPeer, stepSize, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    chunkSize = min(chunkSize, DIVUP(sizePerScclChunk-gridOffset,minChunkSize)*minChunkSize);
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.send(chunkPointer, nelem*count);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recv(chunkPointer, nelem*count);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recvCopySend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceSend(T * chunkPointer, int count) {
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }  

  __device__ void recvReduceCopySend(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL128 : public scclFunction<FUNC, T, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T>
struct LLWrapper {
  const int stepLines;
  const ssize_t chunkSize;
  int nelem;

  ncclLLPrimitives<T, FUNC, 1, 1> prims;

  __device__ LLWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepLines(args->comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS)),
    chunkSize(stepLines * sizeof(uint64_t) / sizeof(T)),
    prims(tid, args->nThreads, recvPeer, sendPeer, stepLines, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.send(chunkPointer, nelem*count);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recv(chunkPointer, nelem*count);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recvCopySend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceSend(T * chunkPointer, int count) {
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }  

  __device__ void recvReduceCopySend(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL : public scclFunction<FUNC, T, LLWrapper<FUNC, T>> {};

// Manually written functions

template<class FUNC, typename T, int UNROLL>
class scclFunctionManual {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      const int tid = threadIdx.x;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for SCCL cross thread synchronization
      const int bid = blockIdx.x;
      const int bdim = blockDim.x;
      struct ncclChannel* channel = comm->channels;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;

      T * thisScratch = (T*)args->scratchbuff;
      int myRank = channel->ring.devUserRanks[0];
      int peer = (myRank > bid) ? bid : bid+1;
      LLWrapper<FUNC,T> prims{args, tid, &peer, &peer, thisInput, channel};

      const ssize_t size = args->coll.count;
      const int sizePerChunk = size/8;
      const ssize_t loopSize = (ssize_t)prims.chunkSize;
      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerChunk; gridOffset += loopSize, iter++) {
        size_t chunkOffset = prims.initIter(sizePerChunk, gridOffset);

        // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
        // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
        const int workIndex = args->index+1;
        volatile struct scclFlag* scclFlags = comm->scclAlgoShared.flags;

        prims.send(thisInput+chunkOffset+peer*sizePerChunk, peer*sizePerChunk+chunkOffset, 1);
        prims.recv(thisScratch+chunkOffset+bid*sizePerChunk, bid*sizePerChunk+chunkOffset, 1);
        if (tid == sync_tid){
          __threadfence();
          uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, 0);
          scclFlags[bid].flag = curFlag;
        }
        if (tid < 7){
          uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, 0);
          while ((scclFlags + tid)->flag < goalFlag){};
        }
        __syncthreads();

        const int nthreads = args->nThreads;
        int upperBound = min(sizePerChunk-gridOffset,prims.chunkSize);
        for (int j = bid*bdim+tid; j < upperBound; j += nthreads*7){
          T t = thisInput[myRank*sizePerChunk+chunkOffset+j];
          for (int i = 0; i < 7; i++){
            T c = thisScratch[i*sizePerChunk+chunkOffset+j];
            t = FUNC()(c, t);
          }
          thisInput[myRank*sizePerChunk+chunkOffset+j] = t;
        }
        __syncthreads();

        if (bid*bdim < sizePerChunk && tid == sync_tid){
          __threadfence();
          uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, 1);
          scclFlags[bid].flag = curFlag;
        }
        if (tid*bdim < sizePerChunk && tid < 7){
          uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, 1);
          while ((scclFlags + tid)->flag < goalFlag){};
        }
        __syncthreads();
        prims.send(thisInput+chunkOffset+myRank*sizePerChunk, myRank*sizePerChunk+chunkOffset, 1);
        prims.recv(thisInput+chunkOffset+peer*sizePerChunk, peer*sizePerChunk+chunkOffset, 1);
      }
    }
};
