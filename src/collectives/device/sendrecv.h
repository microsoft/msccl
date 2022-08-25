/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "collectives.h"
#include "primitives.h"
#include "npkit/npkit.h"

template<typename T, typename RedOp>
struct RunWork<ncclFuncSendRecv, T, RedOp, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE> {
  __device__ __forceinline__ void runSend(const int tid, const int nthreads, const int group, struct ncclWorkElemP2p* args) {
    NPKIT_GPU_SYNC_TIME_SEND(blockIdx.x, tid);

    if (args->peer == ncclShmem.comm.rank) {
      struct ncclWorkElemP2p* recvArgs = args-1;
      if (args->buff != recvArgs->buff) {
        NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_SEND_ENTRY, args->count*sizeof(T));

        ReduceOrCopyMulti<COLL_UNROLL, RedOp, T, 1, 1, 1, 1, 0>(tid, nthreads, nullptr, false, 1, (const T**)&args->buff, 1, (T**)&recvArgs->buff, args->count);

        NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_SEND_EXIT, args->count*sizeof(T));
      }
    } else {
      using Proto = ProtoSimple<1, 1>;
      ssize_t const count = args->count;
      int const chunkSize = args->chunkSize/sizeof(T);
      int const peer = args->peer;
      Primitives<T, RedOp, FanAsymmetric<0, 1>, 1, Proto, 1> prims
        (tid, nthreads, nullptr, &peer, args->buff, nullptr, /*redOpArg(ignored)=*/0, group);

      NPKIT_GPU_SET_CTX_ID(prims);

      ssize_t offset = 0;
      do {
        int nelem = min(chunkSize, count-offset);
        prims.directSend(offset, offset, nelem);
        offset += nelem;
      } while(offset < count);
    }
  }

  __device__ __forceinline__ void runRecv(const int tid, const int nthreads, const int group, struct ncclWorkElemP2p* args) {
    NPKIT_GPU_SYNC_TIME_RECV(blockIdx.x, tid);

    if (args->peer != ncclShmem.comm.rank) {
      using Proto = ProtoSimple<1, 1>;
      ssize_t const count = args->count;
      int const chunkSize = args->chunkSize/sizeof(T);
      int const peer = args->peer;
      Primitives<T, RedOp, FanAsymmetric<1, 0>, 1, Proto, 1> prims
        (tid, nthreads, &peer, nullptr, nullptr, args->buff, /*redOpArg(ignored)=*/0, group);

      NPKIT_GPU_SET_CTX_ID(prims);

      ssize_t offset = 0;
      do {
        int nelem = min(chunkSize, count-offset);

        NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_RECV_ENTRY, nelem*sizeof(T));

        prims.directRecv(offset, nelem);

        NPKIT_GPU_ENTER_EVENT(NPKIT_EVENT_RECV_EXIT, nelem*sizeof(T));

        offset += nelem;
      } while(offset < count);
    }
  }

  __device__ __forceinline__ void run(ncclWork *work) {
    struct ncclWorkElemP2p* args = work->p2pElems;
    int ngroups = args->ngroups;
    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    // This has to work even for groups of 2.5 warps (which is 8 groups, and means 3
    // warps for send, 2 warps for recv).
    // warpStarts were rounded thanks to int division, but for group number we need to round the other way around
    // So we mirror wid then mirror again the group.
    #define NWARPS (NCCL_MAX_NTHREADS/WARP_SIZE)
    int group = ngroups-1- (NWARPS-1-wid) * ngroups / NWARPS;
    args += group;
    if (args->header.type == ncclWorkTypeUnused) return;

    tid -= args->warpStart * WARP_SIZE;
    int nthreads = args->nWarps * WARP_SIZE;
    group |= 1<<16; // Used to select connIndex 1
    if (tid >= nthreads || args->peer == -1) return;
    if ((group%2) == 0) {
      runRecv(tid, nthreads, group, args);
    } else {
      runSend(tid, nthreads, group, args);
    }
  }
};
