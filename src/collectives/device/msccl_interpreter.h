/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

#define MSCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

template<class FUNC, typename T, typename PRIMS_WRAPPER>
class mscclFunction {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      struct mscclAlgorithm* mscclAlgo = &comm->mscclAlgos[args->mscclAlgoIndex];
      const int tid = threadIdx.x;
      const int nThreads = args->nThreads; // last thread is most likely not doing anthing and used for MSCCL cross thread synchronization
      const int bid = blockIdx.x;
      struct mscclThreadBlock* mscclTB = &mscclAlgo->mscclTB[bid];
      const int channelId = mscclTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      T * thisOutput = (T*)args->recvbuff;
      T * thisScratch = (T*)args->scratchbuff;
      int recvPeer = mscclTB->recvpeer;
      int sendPeer = mscclTB->sendpeer;

      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, thisOutput, channel};

      const ssize_t loopSize = (ssize_t)prims.chunkSize;
      const ssize_t size = args->coll.count;
      const ssize_t sizePerMscclChunk = (size*sizeMultiplier)/mscclAlgo->nchunksPerLoop;
      uint32_t mscclMaxAllowedCount = args->mscclMaxAllowedCount;

      // msccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct mscclFlag* mscclFlags = comm->mscclAlgoShared.flags;
      mscclComputeOp_t* mscclComputeOp = &args->mscclComputeOp;

      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerMscclChunk; gridOffset += loopSize, iter++) {
        size_t chunkOffset = prims.initIter(sizePerMscclChunk, gridOffset);
        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        int step = 0;
        for (int i = 0; i < mscclTB->nsteps; i++){
          struct mscclTransfer* msccltran = &mscclTB->transfers[i];
          // first wait if there is a dependence
          int16_t dependentPointer = msccltran->depencePointer;
          int16_t numDependences = msccltran->numDependences;
          if (msccltran->numDependences > 0){
            int index = tid;
	          if (index < numDependences){
              int8_t dependentBid = mscclTB->dependentBid[dependentPointer+index];
              int16_t dependentStep = mscclTB->dependentStep[dependentPointer+index];
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((mscclFlags + dependentBid)->flag < goalFlag){};
            }
            step += msccltran->numDependences-1;
            __syncthreads();
          }

          srcPointer = (msccltran->srcbuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((msccltran->srcbuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstPointer = (msccltran->dstbuffer == MSCCL_INPUT_BUFFER) ? thisInput : ((msccltran->dstbuffer == MSCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          int count = msccltran->count;
          for (int c = 0; c < count; c += mscclMaxAllowedCount) {
            srcoffset = chunkOffset + (ssize_t) (msccltran->srcoffset+c) * sizePerMscclChunk;
            dstoffset = chunkOffset + (ssize_t) (msccltran->dstoffset+c) * sizePerMscclChunk;
            int thisCount = min(mscclMaxAllowedCount, count-c);
            if (msccltran->type == MSCCL_SEND)
              prims.send(srcPointer + srcoffset, dstoffset, thisCount);
            else if (msccltran->type == MSCCL_RECV)
              prims.recv(dstPointer + dstoffset, dstoffset, thisCount);
            else if (msccltran->type == MSCCL_REDUCE) {
              int numReductions = msccltran->numReductions;
              int thisChunkSize = prims.nelem * thisCount;
              dstoffset = chunkOffset + (ssize_t) (msccltran->dstoffset+c) * sizePerMscclChunk;
	      volatile T* s = srcPointer;
	      volatile T* d = dstPointer;
              for (int index = tid; index < thisChunkSize; index += nThreads){
                T o = d[dstoffset + index];
                for (int r = 0; r < numReductions; r++){
                  srcoffset = chunkOffset + (ssize_t) (mscclTB->reductionSrcOffsets[msccltran->reductionPointer+r]+c) * sizePerMscclChunk;
                  T t = s[srcoffset + index];
                  o = FUNC()(o, t);
                }
                d[dstoffset + index] = o;
              }
              step += numReductions-1;
            } else if (msccltran->type == MSCCL_RES_ADD) {
              int thisChunkSize = prims.nelem * thisCount;
              T* resPtr1 = (T*)mscclComputeOp->residualAddOp.residual1;
              T* resPtr2 = (T*)mscclComputeOp->residualAddOp.residual2;
              for (int index = tid; index < thisChunkSize; index += nThreads){
                T r1 = resPtr1[srcoffset+index];
                T r2 = resPtr2[srcoffset+index];
                T o  = dstPointer[dstoffset+index];
                o = FUNC()(o,FUNC()(r1,r2));
                dstPointer[dstoffset+index] = o;
              }
            } else if (msccltran->type == MSCCL_RECV_COPY_SEND)
              prims.recvCopySend(dstPointer + dstoffset, dstoffset, thisCount);
            else if (msccltran->type == MSCCL_RECV_REDUCE_SEND)
              prims.recvReduceSend(srcPointer + srcoffset, thisCount);
            else if (msccltran->type == MSCCL_RECV_REDUCE_COPY_SEND)
              prims.recvReduceCopySend(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
            else if (msccltran->type == MSCCL_RECV_REDUCE_COPY)
              prims.recvReduceCopy(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
            else if (msccltran->type == MSCCL_LOCAL_COPY)
              prims.localCopy(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
            else
              return;
          }
          if (msccltran->has_dependence){
            __syncthreads();
            if (tid == nThreads-1){
              __threadfence();
              uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, step);
              mscclFlags[bid].flag = curFlag;
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

  ncclPrimitives<UNROLL, MSCCL_CHUNKSTEPS/MSCCL_SLICESTEPS, MSCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ SimpleWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
    stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
    chunkSize(stepSize * MSCCL_CHUNKSTEPS),
    prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {}

  __device__ size_t initIter(ssize_t sizePerMscclChunk, ssize_t gridOffset) {
    int realChunkSize = min(chunkSize, sizePerMscclChunk-gridOffset);
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset;
    nelem = min(realChunkSize, sizePerMscclChunk-chunkOffset);
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

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class mscclFunctionSimple : public mscclFunction<FUNC, T, SimpleWrapper<FUNC, T, UNROLL>> {};

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

  __device__ size_t initIter(ssize_t sizePerMscclChunk, ssize_t gridOffset) {
    chunkSize = min(chunkSize, DIVUP(sizePerMscclChunk-gridOffset,minChunkSize)*minChunkSize);
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerMscclChunk-chunkOffset);
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

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class mscclFunctionLL128 : public mscclFunction<FUNC, T, LL128Wrapper<FUNC, T>> {};

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

  __device__ size_t initIter(ssize_t sizePerMscclChunk, ssize_t gridOffset) {
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerMscclChunk-chunkOffset);
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

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class mscclFunctionLL : public mscclFunction<FUNC, T, LLWrapper<FUNC, T>> {};

// Manually written functions

template<class FUNC, typename T, int UNROLL>
class mscclFunctionManual {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      const int tid = threadIdx.x;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for MSCCL cross thread synchronization
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

        // msccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
        // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
        const int workIndex = args->index+1;
        volatile struct mscclFlag* mscclFlags = comm->mscclAlgoShared.flags;

        prims.send(thisInput+chunkOffset+peer*sizePerChunk, peer*sizePerChunk+chunkOffset, 1);
        prims.recv(thisScratch+chunkOffset+bid*sizePerChunk, bid*sizePerChunk+chunkOffset, 1);
        if (tid == sync_tid){
          __threadfence();
          uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, 0);
          mscclFlags[bid].flag = curFlag;
        }
        if (tid < 7){
          uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, 0);
          while ((mscclFlags + tid)->flag < goalFlag){};
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
          mscclFlags[bid].flag = curFlag;
        }
        if (tid*bdim < sizePerChunk && tid < 7){
          uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, 1);
          while ((mscclFlags + tid)->flag < goalFlag){};
        }
        __syncthreads();
        prims.send(thisInput+chunkOffset+myRank*sizePerChunk, myRank*sizePerChunk+chunkOffset, 1);
        prims.recv(thisInput+chunkOffset+peer*sizePerChunk, peer*sizePerChunk+chunkOffset, 1);
      }
    }
};
