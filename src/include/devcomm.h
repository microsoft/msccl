/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_DEVICE_H_
#define NCCL_DEVICE_H_

#include "nccl.h"
#include "align.h"
#include <stdint.h>

<<<<<<< HEAD
#define NCCL_NUM_FUNCTIONS 7 // SendRecv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncAllToAll, ncclFuncCustomCollective, ncclFuncSendRecv} ncclFunc_t;
=======
#define NCCL_NUM_FUNCTIONS 5 // Send/Recv not included for now
typedef enum { ncclFuncBroadcast, ncclFuncReduce, ncclFuncAllGather, ncclFuncReduceScatter, ncclFuncAllReduce, ncclFuncSendRecv, ncclFuncSend, ncclFuncRecv, ncclNumFuncs} ncclFunc_t;
>>>>>>> upstream/master
extern const char* ncclFuncStr[NCCL_NUM_FUNCTIONS];

#define NCCL_NUM_ALGORITHMS 4 // Tree/Ring/MSCCL/CollNet
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_MSCCL 2
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

#define NCCL_LL128_SHMEM_ELEMS_PER_THREAD 8
#define NCCL_LL128_SHMEM_SIZE (NCCL_LL128_SHMEM_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS)

#define NCCL_DIRECT_WRITE 0x01
#define NCCL_DIRECT_READ  0x02
#define NCCL_DIRECT_NIC   0x04
#define NCCL_IPC_WRITE    0x08
#define NCCL_IPC_READ     0x10

struct ncclConnInfo {
  // Regular comm mechanism
  char *buffs[NCCL_NUM_PROTOCOLS]; // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv

  int direct;         // Direct communication
  int shared;         // Buffers are shared
  void **ptrExchange; // Pointer exchange for direct communication
  uint64_t* redOpArgExchange; // PreOp scaler exchange for direct pull case

  int *sizesFifo;     // Sizes fifo from GPU to proxy
  int *offsFifo;      // Buffer fifo from proxy to GPU

  uint64_t step;      // Keep where we are
  uint64_t llLastCleaning;
};

struct ncclProxyConnector {
  int rank;
  int localRank;
  struct ncclProxyConnection* connection;
  struct ncclComm* comm;
};

struct ncclConnector {
  int connected;
  struct ncclProxyConnector proxyConn;
  struct ncclTransportComm* transportComm;
  void* transportResources;
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

  int index; // This rank's index in the ring
};

#define MSCCL_MAX_NUM_STEPS 256
#define MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL 32
#define MSCCL_MAX_NUM_THREAD_BLOCKS (108*2) // set this to 108 which is the number of SMs on A100

static_assert(MAXCHANNELS*MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL >= MSCCL_MAX_NUM_THREAD_BLOCKS);
static_assert(MSCCL_MAX_NUM_STEPS <= 256, "MSCCL interpreter doesn't allow for more than nthreads dependences");

#define MSCCL_INPUT_BUFFER 0
#define MSCCL_OUTPUT_BUFFER 1
#define MSCCL_SCRATCH_BUFFER 2

#define MSCCL_SEND 0
#define MSCCL_RECV 1
#define MSCCL_RECV_COPY_SEND 2
#define MSCCL_RECV_REDUCE_SEND 3
#define MSCCL_RECV_REDUCE_COPY 4
#define MSCCL_RECV_REDUCE_COPY_SEND 5
#define MSCCL_LOCAL_COPY 6
#define MSCCL_REDUCE 7
#define MSCCL_RES_ADD 8

// TODO: compress this by a lot!
struct mscclTransfer {
  int16_t srcoffset;
  int16_t dstoffset;
  uint8_t srcbuffer; // follow MSCCL_THIS_INPUT/MSCCL_THIS_OUTPUT macros
  uint8_t dstbuffer; // follow MSCCL_THIS_INPUT/MSCCL_THIS_OUTPUT macros
  int16_t depencePointer; // index to the first dependence
  int16_t numDependences; // depencePointer+numDependences indicate the last dependence
  int8_t has_dependence;
  int16_t numReductions; // number of reductions with the same dst
  int16_t reductionPointer; // where the reduction starts
  uint8_t type;
  uint8_t count;
};

struct mscclThreadBlock {
  int16_t sendpeer;
  int16_t recvpeer;
  uint16_t nsteps;
  int8_t channelId; // associated channel. -1 indicates a threadblock with only local copies
  // step is used to index into this array. transfers[step] is the addr to transfer.
  int8_t dependentBid[MSCCL_MAX_NUM_STEPS]; // -1 if not dependent on any threadblock
  int16_t dependentStep[MSCCL_MAX_NUM_STEPS];
  int16_t reductionSrcOffsets[MSCCL_MAX_NUM_STEPS]; // in case there are multiple reductions with the same dstwewqwqew
  struct mscclTransfer transfers[MSCCL_MAX_NUM_STEPS];
};

#define MSCCL_MAX_COUNT 72

struct mscclChannelInfo {
  int sendPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // nchunksForSendPeer[i][j] represents the number of times chunks are sent in counts of j-1 for threadblock i. we do not keep counts of 0.
  int nchunksForSendPeer[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][MSCCL_MAX_COUNT];
  int nsendPeers;
  int recvPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  int nchunksForRecvPeer[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][MSCCL_MAX_COUNT];
  int nrecvPeers;
  int nBlocksForChannel;
};

struct mscclFlag {
  uint64_t flag;
  uint64_t align[3]; // To avoid false sharing
};

// gpuId is the one that is in comm->rank
struct mscclAlgorithm {
#define MSCCL_MAX_ALGO_NAME 63
  // name of the algorithm in the XML
  char name[MSCCL_MAX_ALGO_NAME+1];
  // a flag to specify if the MSCCL algorithm is a valid one
  bool isValid;
  // the type of collective this algorithm is
  ncclFunc_t collectiveType;
  // inPlace collective
  int inPlace;
  // number of gpus in the group
  int ngpus;
  // max(#chunks in input, #chunks in output)
  int nchunksPerLoop;
  // the protocol that the algorithm needs to use
  int protocol;
  // the range of size in which this algorithm is performant
  int64_t minBytes; int64_t maxBytes;
  // bid is used as an index into this array
  struct mscclThreadBlock mscclTB[MAXCHANNELS*MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // number of channels needed by MSCCL algorithm
  int nChannels;
  // the arrays in this struct can be inferred from mscclTB. they are created to use NCCL API easily
  struct mscclChannelInfo mscclChannels[MAXCHANNELS];
  // number of scratch chunks that MSCCL will use
  int nScratchChunks;
  //Reduction Operator. If the algorithm performs reduction it will specify the reduction operator.
  //If the algorithm do not perform reduction, its reduction operator is considered as ncclSum.
  ncclRedOp_t redOp;
};

struct mscclAlgorithmShared {
  // allocate enough MSCCL flags (MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL * MAXCHANNELS) to synchronize across thread blocks
  struct mscclFlag* flags;
  // this flag is used to indicate we have we have looped around the channels work queue. Once that happens, the flags need to be reset.
  int flagsNeedReset;
  // declaration for scratchBuffer. This is only to be accessed by the host
  size_t scratchBufferSize;
  void* scratchBuffer;
};

#define NCCL_MAX_TREE_ARITY 3
struct ncclTree {
  int depth;
  int up;
  int down[NCCL_MAX_TREE_ARITY];
};

#define NCCL_MAX_DIRECT_ARITY 7
struct ncclDirect {
  int depth;
  int out;
  int nHeads;
  int headRank;
  int shift;
  int up[NCCL_MAX_DIRECT_ARITY];
  int down[NCCL_MAX_DIRECT_ARITY];
};

#define NCCL_MAX_CONNS 2
struct ncclPeer {
  struct ncclConnector send[NCCL_MAX_CONNS];
  struct ncclConnector recv[NCCL_MAX_CONNS];
};

struct ncclDevComm;

/* ncclWork is to be a power of two, currently 8x64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclWorkElem. */
#define NCCL_WORK_SIZE 512

enum ncclWorkElemType : uint8_t {
   ncclWorkTypeUnused=0,
   ncclWorkTypeColl=1,
   ncclWorkTypeP2p=2,
   ncclWorkTypeRegColl=3
};
enum ncclWorkElemSubType : uint8_t {
  ncclWorkSubTypeUnused =0,
  ncclWorkSubTypeSend,
  ncclWorkSubTypeRecv
};

struct ncclWorkElemHeader {
  uint16_t funcIndex;
<<<<<<< HEAD
  uint16_t index;

  // in MSCCL algorithms, only one workelem at a time is allowed.
  uint8_t active;
  uint32_t mscclMaxAllowedCount; // this is used in mscclAlgorithm to find the maximum number of counts that can be sent at the same time.
  int mscclAlgoIndex; // taken from info->mscclAlgoIndex
  mscclComputeOp_t mscclComputeOp; // taken from info->mscclComputeOp
=======
  enum ncclWorkElemType type;
  unsigned nWarps:5;
  unsigned isLast:1;
};

struct ncclWorkElem {
  struct ncclWorkElemHeader header;
  uint8_t regUsed;
  uint8_t direct;
  uint8_t redOpArgIsPtr;
>>>>>>> upstream/master

  const void * sendbuff;
  void * recvbuff;
  void * scratchbuff;

<<<<<<< HEAD
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
    int8_t align[64];
  };
=======
  size_t count;
  size_t lastChunkSize;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint64_t redOpArg;
  uint64_t pad;
};
static_assert(NCCL_WORK_SIZE % sizeof(struct ncclWorkElem) == 0, "ncclWorkElem size must be a multiple of ncclWork size");

struct ncclWorkElemP2p {
  struct ncclWorkElemHeader header;
  int32_t peer;
  void* buff;
  size_t count;
  int chunkSize;
  uint8_t ngroups;
  uint8_t warpStart;
  uint8_t nWarps;
  enum ncclWorkElemSubType subType;
};
static_assert(NCCL_WORK_SIZE % sizeof(struct ncclWorkElemP2p) == 0, "ncclWorkElemP2p size must be a multiple of ncclWork size");

struct ncclWorkElemReg {
  struct ncclWorkElem elem;
  void* dnInputs[NCCL_MAX_DIRECT_ARITY+1];
  void* dnOutputs[NCCL_MAX_DIRECT_ARITY+1];
  void* upOutputs[NCCL_MAX_DIRECT_ARITY+1];
>>>>>>> upstream/master
};
static_assert(NCCL_WORK_SIZE % sizeof(struct ncclWorkElemReg) == 0, "ncclWork size must be a multiple of ncclWorkElemReg size");
static_assert(sizeof(struct ncclWorkElemReg) % sizeof(struct ncclWorkElem) == 0, "ncclWorkElemReg size must be a multiple of ncclWorkElem size");

#define NCCL_MAX_WORK_ELEMENTS (NCCL_WORK_SIZE/sizeof(struct ncclWorkElem))
#define NCCL_MAX_WORK_ELEMENTS_P2P (NCCL_WORK_SIZE/sizeof(struct ncclWorkElemP2p))
#define NCCL_MAX_WORK_ELEMENTS_REG (NCCL_WORK_SIZE/sizeof(struct ncclWorkElemReg))
// Number of named barriers supported by CUDA
#define NCCL_MAX_GROUPS 16

struct ncclWork {
  union {
    char pad[NCCL_WORK_SIZE];
    struct ncclWorkElemHeader header;
    struct ncclWorkElem elems[NCCL_MAX_WORK_ELEMENTS];
    struct ncclWorkElemP2p p2pElems[NCCL_MAX_WORK_ELEMENTS_P2P];
    struct ncclWorkElemReg regElems[NCCL_MAX_WORK_ELEMENTS_REG];
  };
};
<<<<<<< HEAD
static_assert(sizeof(struct ncclWorkElem) == (128), "ncclWorkElem must have a pow2 size");
=======

static_assert(sizeof(struct ncclWork) == NCCL_WORK_SIZE, "ncclWork size needs to be well aligned");
>>>>>>> upstream/master

struct ncclChannel {
  union {
    struct {
      struct ncclRing ring;
      struct ncclTree tree;
      struct ncclDirect collTree;

      int id;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      struct ncclWork* workFifo;
      int workCount;
      size_t totalSize;
      uint64_t workFifoTail; // Only used by CPU
      uint16_t index;        // Only used by GPU

      // GDRCOPY support
      struct ncclWork* workFifoGdr;
      struct ncclWork* workFifoDev;
      void* gdrMemDesc;
    };
    int data[0x80];
  };
};
static_assert(sizeof(struct ncclChannel) == 0x80*sizeof(int), "ncclChannel must have a pow2 size");

#define MSCCL_MAX_NUM_ALGOS 4
struct ncclDevComm {
  int rank;
  int nRanks;
  int buffSizes[NCCL_NUM_PROTOCOLS];

  // MSCCL related elements
  int numberOfMSCCLAlgorithms;
  struct mscclAlgorithm mscclAlgos[MSCCL_MAX_NUM_ALGOS];
  struct mscclAlgorithmShared mscclAlgoShared;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Channels, device side
  struct ncclChannel* channels;
};

<<<<<<< HEAD
cudaError_t strideMemcpyAsync(void* dst, const void* src, const size_t size, const int height, const int width, cudaStream_t stream);
=======
struct ncclDevCommAndChannels {
  ncclDevComm comm;
  ncclChannel channels[MAXCHANNELS];
};
>>>>>>> upstream/master

#endif
