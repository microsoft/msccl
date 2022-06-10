/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INFO_H_
#define NCCL_INFO_H_

#include "nccl.h"
#include "devcomm.h"
#include "collectives.h"

typedef enum : uint8_t {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown,
<<<<<<< HEAD
  ncclPatternCollTreeUp,
  ncclPatternCollTreeDown,
  ncclPatternMsccl
=======
  ncclPatternCollTreeUpDown,
  ncclPatternSend,
  ncclPatternRecv
>>>>>>> upstream/master
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  int inplace; // needed for msccl
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  ncclDevRedOpFull opFull;
  int algorithm;
  int mscclAlgoIndex; // Used to indentify MSCCL algorithm
  mscclComputeOp_t mscclComputeOp; // msccl operation for custom compute
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
  int chunkSize;
  int channelId;
  
  // MSCCL scratch buffer is passed as an arg
  // this scratchBuffer can be accessed on the device. The management of this memory is on mscclAlgorithm
  void* scratchbuff;
};

#endif
