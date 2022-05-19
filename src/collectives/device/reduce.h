/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
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
      const int chunkSize = stepSize * REDUCE_CHUNKSTEPS;
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = ring->devUserRanks[0];
      const int prevRank = ring->devUserRanks[nranks-1];
      const int root = args->coll.root;

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      ncclPrimitives<UNROLL, REDUCE_CHUNKSTEPS/REDUCE_SLICESTEPS, REDUCE_SLICESTEPS, T, 1, 1, 0, FUNC>
        prims(tid, nthreads, &ring->prev, &ring->next, NULL, stepSize, channel, comm, ncclShmem->ptrs, 0);

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        int realChunkSize = min(chunkSize, DIVUP(size-gridOffset,nChannels));
        ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
        ssize_t offset = gridOffset + bid*realChunkSize;
        int nelem = min(realChunkSize, size-offset);
        if (prevRank == root) {
          prims.send(thisInput+offset, nelem);
        } else if (rank == root) {
          prims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else {
          prims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
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
      const int rank = comm->rank;
      const int prevRank = ring->devUserRanks[nranks-1];
      const int root = args->coll.root;

      ncclLLPrimitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepLines, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        if (size-gridOffset < loopSize) {
          chunkSize = args->coll.lastChunkSize;
        }
        ssize_t offset = gridOffset + bid*chunkSize;

        int nelem = min(chunkSize, size-offset);
        if (prevRank == root) {
          LLprims.send(thisInput+offset, nelem);
        } else if (rank == root) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }
};

#include "prims_ll128.h"
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncReduce, NCCL_ALGO_RING, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
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
      const ssize_t minChunkSize = (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*nthreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T));
      const int nranks = comm->nRanks;
      const ssize_t loopSize = nChannels*chunkSize;
      const ssize_t size = args->coll.count;
      const int rank = comm->rank;
      const int prevRank = ring->devUserRanks[nranks-1];
      const int root = args->coll.root;

      ncclLL128Primitives<T, FUNC, 1, 1> LLprims(tid, nthreads, &ring->prev, &ring->next, stepSize, channel, comm);

      // Compute pointers
      const T * __restrict__ thisInput = (const T*)args->sendbuff;
      T * __restrict__ thisOutput = (T*)args->recvbuff;

      for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize) {
        chunkSize = min(DIVUP(size-gridOffset, nChannels*minChunkSize)*minChunkSize, chunkSize);
        ssize_t offset = gridOffset + bid*chunkSize;

        int nelem = min(chunkSize, size-offset);
        if (prevRank == root) {
          LLprims.send(thisInput+offset, nelem);
        } else if (rank == root) {
          LLprims.recvReduceCopy(thisInput+offset, thisOutput+offset, nelem);
        } else {
          LLprims.recvReduceSend(thisInput+offset, nelem);
        }
      }
    }
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncReduce, NCCL_ALGO_TREE, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncReduce, NCCL_ALGO_COLLNET, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class REDOP, typename T, int UNROLL>
class ncclFunction<ncclFuncReduce, NCCL_ALGO_MSCCL, PROTO, REDOP, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};
