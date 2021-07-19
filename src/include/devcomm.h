/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "align.h"
#include <stdint.h>

#define NCCL_NUM_FUNCTIONS 6 // SendRecv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncAllToAll, ncclFuncSendRecv} ncclFunc_t;
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

#define NCCL_NUM_ALGORITHMS 4 // Tree/Ring/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_SCCL 2
#define NCCL_ALGO_COLLNET 3
extern const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS];

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
extern const char* ncclProtoStr[NCCL_NUM_PROTOCOLS];

#define NCCL_MAX_OPS 2048
#define NCCL_STEPS 8
union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

#define WARP_SIZE 32
#define MAXCHANNELS 32
#define NCCL_MAX_NTHREADS 640
#define NCCL_SIMPLE_MAX_NTHREADS 512
#define NCCL_LL_MAX_NTHREADS 512
#define NCCL_LL_LINES_PER_THREAD 8
#ifdef TEST_LL_CLEANUP
#define NCCL_LL_CLEAN_MASK 0x078 // Set to 0x100 to disable cleanup
#define NCCL_LL_FLAG_MAX   0x100
#define NCCL_LL_FLAG(a) ((uint32_t)((a) % NCCL_LL_FLAG_MAX))
#else
#define NCCL_LL_CLEAN_MASK 0x7ffffff8
#define NCCL_LL_FLAG(a) ((uint32_t)(a))
#endif
// Make sure the clean mask will last for at least NCCL_NSTEPS
static_assert(NCCL_LL_CLEAN_MASK % NCCL_STEPS == 0, "Invalid NCCL_LL_CLEAN_MASK value");

#define NCCL_LL128_LINESIZE 128
#define NCCL_LL128_LINEELEMS (NCCL_LL128_LINESIZE/sizeof(uint64_t))
#define NCCL_LL128_DATAELEMS (NCCL_LL128_LINEELEMS-1)

#define NCCL_LL128_MAX_NTHREADS 640
#define NCCL_LL128_ELEMS_PER_THREAD 120

// Receiving from up to 3 sources is more compute intensive than sending
// to 3 dests. Use 70% for reduce and 30% for bcast.
#define NCCL_LL128_SPLIT(nt) ((nt*7/(10*32))*32)

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_DIRECT_GPU 0x01
#define NCCL_DIRECT_NIC 0x10

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  void* *ptrsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct ncclConnector {
  int connected;
  struct ncclProxyArgs *proxyAppend;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
  struct ncclComm *comm;
};

struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
  int* devUserRanks;
};

#define SCCL_MAX_NUM_STEPS 512
#define SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL 32

#define SCCL_INPUT_BUFFER 0
#define SCCL_OUTPUT_BUFFER 1
#define SCCL_SCRATCH_BUFFER 2
#define SCCL_ARG_BUFFERS_BEGIN 3
#define SCCL_ARG_BUFFERS_END (SCCL_ARG_BUFFERS_BEGIN + SCCL_NUM_ARG_BUFFS)

#define SCCL_SEND 0
#define SCCL_RECV 1
#define SCCL_RECV_COPY_SEND 2
#define SCCL_RECV_REDUCE_SEND 3
#define SCCL_RECV_REDUCE_COPY 4
#define SCCL_NO_OP 5
#define SCCL_BINARY_OP_BEGIN 6
#define SCCL_ADD (SCCL_BINARY_OP_BEGIN + 0)
#define SCCL_SUB (SCCL_BINARY_OP_BEGIN + 1)
#define SCCL_MUL (SCCL_BINARY_OP_BEGIN + 2)
#define SCCL_MIN (SCCL_BINARY_OP_BEGIN + 3)
#define SCCL_MAX (SCCL_BINARY_OP_BEGIN + 4)
#define SCCL_BINARY_OP_END (SCCL_MAX)
#define SCCL_ISQRT 11

// TODO: compress this by a lot!
struct scclTransfer {
  int16_t srcoffset;
  int16_t dstoffset;
  int16_t src2offset;
  uint8_t srcbuffer; // follow SCCL_THIS_INPUT/SCCL_THIS_OUTPUT macros
  uint8_t dstbuffer; // follow SCCL_THIS_INPUT/SCCL_THIS_OUTPUT macros
  uint8_t src2buffer; // follow SCCL_THIS_INPUT/SCCL_THIS_OUTPUT macros
  int8_t dependentBid; // -1 if not dependent on any threadblock
  int16_t dependentStep;
  int8_t has_dependence;
  uint8_t type;
  uint8_t count;
};

struct scclThreadBlock {
  int8_t sendpeer;
  int8_t recvpeer;
  uint16_t nsteps;
  uint8_t channelId; // associated channel
  uint16_t rid; // relative id of this thread block to the channel
  // step is used to index into this array. transfers[step] is the addr to transfer.
  struct scclTransfer transfers[SCCL_MAX_NUM_STEPS];
};

#define SCCL_MAX_COUNT 16

struct scclChannelInfo {
  int sendPeers[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // nchunksForSendPeer[i][j] represents the number of times chunks are sent in counts of j-1 for threadblock i. we do not keep counts of 0.
  int nchunksForSendPeer[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][SCCL_MAX_COUNT];
  int nsendPeers;
  int recvPeers[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  int nchunksForRecvPeer[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][SCCL_MAX_COUNT];
  int nrecvPeers;
  int nBlocksForChannel;
};

struct scclFlag {
  uint64_t flag;
  uint64_t align[3]; // To avoid false sharing
};

// gpuId is the one that is in comm->rank
struct scclAlgorithm {
  // max(#chunks in input, #chunks in output)
  int nchunksPerLoop;
  // the protocol that the algorithm needs to use
  int protocol;
  // total number of threadblocks needed by SCCL algorithm
  int nBlocks; // TODO could be removed
  // bid is used as an index into this array
  struct scclThreadBlock scclTB[MAXCHANNELS*SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // number of channels needed by SCCL algorithm
  int nChannels;
  // the arrays in this struct can be inferred from scclTB. they are created to use NCCL API easily
  struct scclChannelInfo scclChannels[MAXCHANNELS];
  // number of scratch chunks that SCCL will use
  int nScratchChunks;
  // declaration for scratchBuffer. This is only to be accessed by the host
  size_t scratchBufferSize;
  void* scratchBuffer;

  // allocate enough SCCL flags (SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL * MAXCHANNELS) to synchronize across thread blocks
  struct scclFlag* flags;
  // this flag is used to indicate we have we have looped around the channels work queue. Once that happens, the flags need to be reset.
  int flagsNeedReset;
};

#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

struct ncclPeer {
  struct ncclConnector send;
  struct ncclConnector recv;
};

struct ncclDevComm;

#define NCCL_MAX_WORK_ELEMENTS 8
#define NCCL_MAX_GROUPS (NCCL_MAX_WORK_ELEMENTS*2)

/* ncclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclWorkElem. */
struct ncclWorkElem {
  // Header
  struct ncclDevComm* comm;
  uint16_t nThreads;
  uint16_t funcIndex;
  uint16_t index;
  // in SCCL algorithms, ncclWorkElem.active element from workFifo is replicated for for all other thread blocks
  uint8_t active[SCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  uint8_t nActives; // if it is a sccl algorithm, it must be set to associated channel number of thread blocks. if not a sccl algorithm, it is 1.
  uint8_t isScclAlgorithm; // right now, 0 indicates not a sccl algorithm and 1 indicates it is. In future versions, this will be the index into arrays of scclAlgorithms.
  uint32_t scclMaxAllowedCount; // this is used in scclAlgorithm to find the maximum number of counts that can be sent at the same time.

  const void * sendbuff;
  void * recvbuff;
  void * scratchbuff;
  argBuffs_t argbuffs;

  // Op-specific fields.
  union {
    struct {
      size_t count;
      size_t lastChunkSize;
      uint32_t root;
      uint8_t bid;
      uint8_t nChannels;
    } coll;
    struct {
      size_t sendCount;
      size_t recvCount;
      int32_t delta;
      uint16_t nThreads;
    } p2p;
    uint64_t align[6];
  };
};
struct ncclWork {
  struct ncclWorkElem elems[NCCL_MAX_WORK_ELEMENTS];
};
static_assert(sizeof(struct ncclWorkElem) == (0x40*sizeof(int)), "ncclWorkElem must have a pow2 size");

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;
      struct ncclTree collTree;

      int id;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclWork* workFifo;
      int workCount;
      uint64_t workFifoTail; // Only used by CPU
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];
  struct scclAlgorithm scclAlgo;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct ncclChannel* channels;
};

#endif
