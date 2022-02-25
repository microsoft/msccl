/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclSynchronize, int workIndex, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclSynchronize(int workIndex, ncclComm* comm, cudaStream_t stream) {
  return synchronize(workIndex, comm, stream);
}
