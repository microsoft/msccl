/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "npkit/npkit.h"

#define MSCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
  MSCCL_MAX_ITER*MSCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * MSCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

namespace {
  // a copy of the volatile load/store from prims_ll
  template<typename U>
  __device__ static U load(U *src) {
    union {
      U elt;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    if(sizeof(U) == 1)
      asm("ld.volatile.global.b8 %0,[%1];" : "=r"(u4) : "l"(src));
    else if(sizeof(U) == 2)
      asm("ld.volatile.global.b16 %0,[%1];" : "=h"(u2) : "l"(src));
    else if(sizeof(U) == 4)
      asm("ld.volatile.global.b32 %0,[%1];" : "=r"(u4) : "l"(src));
    else
      asm("ld.volatile.global.b64 %0,[%1];" : "=l"(u8) : "l"(src));
    return elt;
  }

  template<typename U>
  __device__ static void store(U *dst, U val) {
    union {
      U elt;
      uint16_t u2;
      uint32_t u4;
      uint64_t u8;
    };
    elt = val;
    if(sizeof(U) == 1)
      asm("st.volatile.global.b8 [%0],%1;" :: "l"(dst), "r"(u4));
    else if(sizeof(U) == 2)
      asm("st.volatile.global.b16 [%0],%1;" :: "l"(dst), "h"(u2));
    else if(sizeof(U) == 4)
      asm("st.volatile.global.b32 [%0],%1;" :: "l"(dst), "r"(u4));
    else
      asm("st.volatile.global.b64 [%0],%1;" :: "l"(dst), "l"(u8));
  }

  inline __device__ void barrier(int nthreads) {
    asm volatile ("bar.sync %1, %0;" :: "r"(nthreads), "r"(15));
  }

  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runInterpreterOld(ncclWorkElem *args, int sizeMultiplier) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x; //args->header.nWarps*WARP_SIZE;
    const int bid = blockIdx.x;
    // struct mscclThreadBlock* mscclTB = &ncclShmem.mscclShmem.mscclTB;

    // User pointers for primitives
    T* thisInput = (T*)args->sendbuff;
    T* thisOutput = (T*)args->recvbuff;
    // int recvPeer = mscclTB->recvpeer;
    // int sendPeer = mscclTB->sendpeer;
    int rank = ncclShmem.comm.rank;
    int peer = -1;
    if (bid > 0)
      peer = (rank > bid-1) ? (bid-1) : bid;

    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? MSCCL_CHUNKSTEPS : 1));
    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    NPKIT_GPU_SYNC_TIME(bid, tid);
    int m1 = -1;
    int others[7*8] = {
	    1,2,3,4,5,6,7,
	    0,2,3,4,5,6,7,
	    0,1,3,4,5,6,7,
	    0,1,2,4,5,6,7,
	    0,1,2,3,5,6,7,
	    0,1,2,3,4,6,7,
	    0,1,2,3,4,5,7,
	    0,1,2,3,4,5,6
    };
    int* myNghrs = &others[rank*7];
    
    const ssize_t size = args->count;
    const ssize_t sizePerMscclChunk = size/8;

    const int64_t workIndex = args->mscclWork.workIndex;
    volatile struct mscclFlag* mscclFlags = args->mscclWork.flags; //ncclShmem.mscclShmem.flags;

    ssize_t gridOffset = 0;
    ssize_t realChunkSize;
    realChunkSize = min(chunkSize, divUp(sizePerMscclChunk-gridOffset, minChunkSize)*minChunkSize);
    realChunkSize = int(realChunkSize);
    int nelem = min(realChunkSize, sizePerMscclChunk-gridOffset);

    RedOp redFn(args->redOpArg);
    if (bid == rank){
      Primitives<T, RedOp, FanSymmetric<7>, 0, Proto, 0> prims
        (tid, nthreads, myNghrs, myNghrs, thisInput, thisOutput, args->redOpArg);
      prims.setDataPtrs(thisInput, thisInput);
      prims.recvReduceCopy(rank*nelem, rank*nelem, nelem);
      prims.send(rank*nelem, nelem);
    } else {
      Primitives<T, RedOp, FanSymmetric<1>, 0, Proto, 0> prims
        (tid, nthreads, &bid, &bid, thisInput, thisOutput, args->redOpArg);
      prims.setDataPtrs(thisInput, thisInput);
      prims.send(bid*nelem, nelem);
      prims.recv(bid*nelem, nelem);
    }
  }

  // -----------------------
  template<typename T, typename RedOp, typename Proto>
  __device__ __forceinline__ void runInterpreter(ncclWorkElem *args, int sizeMultiplier) {
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x; //args->header.nWarps*WARP_SIZE;
    const int bid = blockIdx.x;
    // struct mscclThreadBlock* mscclTB = &ncclShmem.mscclShmem.mscclTB;

    // User pointers for primitives
    T* thisInput = (T*)args->sendbuff;
    T* thisOutput = (T*)args->recvbuff;
    // int recvPeer = mscclTB->recvpeer;
    // int sendPeer = mscclTB->sendpeer;
    int rank = ncclShmem.comm.rank;
    int peer = -1;
    if (bid > 0)
      peer = (rank > bid-1) ? (bid-1) : bid;

    const ssize_t chunkSize = int(Proto::calcBytePerStep()/sizeof(T) * (Proto::Id == NCCL_PROTO_SIMPLE ? MSCCL_CHUNKSTEPS : 1));
    int minChunkSize;
    if (Proto::Id == NCCL_PROTO_LL)
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T));
    if (Proto::Id == NCCL_PROTO_LL128) {
      // We should not need the final /2 but it makes performance much, much smoother. Might be a bug somewhere.
      minChunkSize = nthreads*(Proto::calcBytePerGrain()/sizeof(T))/2;
    }

    NPKIT_GPU_SYNC_TIME(bid, tid);

    RedOp redFn(args->redOpArg);
    Primitives<T, RedOp, FanSymmetric<1>, 1, Proto, 0> prims
      (tid, nthreads, &peer, &peer, thisInput, thisOutput, args->redOpArg);

    NPKIT_GPU_SET_CTX_ID(prims);

    const ssize_t size = args->count;
    const ssize_t sizePerMscclChunk = size/64;
    // uint16_t mscclMaxAllowedCount = args->mscclWork.mscclMaxAllowedCount;
    // int8_t needsFence = ncclShmem.mscclShmem.needsFence;


    // msccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
    // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
    const int64_t workIndex = args->mscclWork.workIndex;
    volatile struct mscclFlag* mscclFlags = args->mscclWork.flags; //ncclShmem.mscclShmem.flags;
    // return; // 3.04
    // for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerMscclChunk; gridOffset += chunkSize, iter++) {
    {
      ssize_t gridOffset = 0;
      ssize_t realChunkSize;
      realChunkSize = min(chunkSize, divUp(sizePerMscclChunk-gridOffset, minChunkSize)*minChunkSize);
      realChunkSize = int(realChunkSize);
      int nelem = min(realChunkSize, sizePerMscclChunk-gridOffset);
      
      // ssize_t srcoffset, dstoffset;
      // T* srcPointer, * dstPointer;
      if (bid > 0){
        prims.setDataPtrs(thisInput, thisInput);
        prims.send(peer*nelem*8, 8*nelem);
        prims.recv(peer*nelem*8, 8*nelem);
        if (tid == nthreads-1){
          mscclFlags[bid].flag = (uint64_t) COMPUTE_FLAG(workIndex, 0, 0);
        }
      }
      if (tid < 7){
        uint64_t goalFlag = COMPUTE_FLAG(workIndex, 0, 0);
        while ((mscclFlags + tid+1)->flag < goalFlag){};
      }
      __syncthreads();
      for (int index = tid; index < nelem; index += nthreads){
        T o = load(thisInput+rank*nelem*8+bid*nelem+index);
        for (int r = 0; r < 7; r++){
          int nghr = (rank > r) ? r : r+1;
          T t = load(thisInput+nghr*nelem*8+bid*nelem+index);
          o = redFn(o, t);
        }
        store(thisInput+rank*nelem*8+bid*nelem+index, o);
      }
      __syncthreads();
      if (tid == nthreads-1){
        mscclFlags[bid].flag = (uint64_t) COMPUTE_FLAG(workIndex, 0, 1);
      }
      if (tid < 8){
        uint64_t goalFlag = COMPUTE_FLAG(workIndex, 0, 1);
        while ((mscclFlags + tid)->flag < goalFlag){};
      }
      __syncthreads();
      if (bid > 0){
        // prims.setDataPtrs(thisInput, thisInput);
        prims.send(rank*nelem*8, 8*nelem);
        prims.recv(peer*nelem*8, 8*nelem);
      }
    }
  }
}
