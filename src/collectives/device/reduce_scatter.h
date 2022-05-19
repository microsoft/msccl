/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "msccl_interpreter.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
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
      const int chunkSize = stepSize * REDUCESCATTER_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      ncclPrimitives<UNROLL, REDUCESCATTER_CHUNKSTEPS/REDUCESCATTER_SLICESTEPS, REDUCESCATTER_SLICESTEPS, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t chunkOffset = gridOffset + bid*realChunkSize;

        /////////////// begin ReduceScatter steps ///////////////
        ssize_t offset;
        int nelem = min(realChunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[nranks-1];
        offset = chunkOffset + rankDest * size;

        prims.send(thisInput+offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          prims.recvReduceSend(thisInput+offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final result
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        prims.recvReduceCopy(thisInput+offset, thisOutput+chunkOffset, nelem);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
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

        /////////////// begin ReduceScatter steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[nranks-1];
        offset = chunkOffset + rankDest * size;

        LLprims.send(thisInput+offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvReduceSend(thisInput+offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        LLprims.recvReduceCopy(thisInput+offset, thisOutput+chunkOffset, nelem);
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
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

        /////////////// begin ReduceScatter steps ///////////////
        ssize_t offset;
        int nelem = min(chunkSize, size-chunkOffset);
        int rankDest;

        // step 0: push data to next GPU
        rankDest = ring->devUserRanks[nranks-1];
        offset = chunkOffset + rankDest * size;

        LLprims.send(thisInput+offset, nelem);

        // k-2 steps: reduce and copy to next GPU
        for (int j=2; j<nranks; ++j) {
          rankDest = ring->devUserRanks[nranks-j];
          offset = chunkOffset + rankDest * size;

          LLprims.recvReduceSend(thisInput+offset, nelem);
        }

        // step k-1: reduce this buffer and data, which will produce the final
        // result that we store in this data
        rankDest = ring->devUserRanks[0];
        offset = chunkOffset + rankDest * size;

        LLprims.recvReduceCopy(thisInput+offset, thisOutput+chunkOffset, nelem);
      }
    }
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_TREE, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_COLLNET, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_MSCCL, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionSimple<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, args->comm->nRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_MSCCL, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL128<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, args->comm->nRanks);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduceScatter, NCCL_ALGO_MSCCL, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, args->comm->nRanks);
    }
};
