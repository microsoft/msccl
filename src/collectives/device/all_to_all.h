/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "msccl_interpreter.h"

template<int ALGO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllToAll, ALGO, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionSimple<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, args->comm->nRanks);
    }
};

template<int ALGO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllToAll, ALGO, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL128<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, args->comm->nRanks);
    }
};

template<int ALGO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllToAll, ALGO, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, args->comm->nRanks);
    }
};
