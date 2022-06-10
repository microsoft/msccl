/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "msccl_interpreter.h"
 
template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_SIMPLE> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionSimple<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, 1);
    }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_LL128> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL128<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, 1);
    }
};

template<typename T, typename RedOp>
struct RunWorkElement<ncclFuncCustomCollective, T, RedOp, NCCL_ALGO_MSCCL, NCCL_PROTO_LL> {
    public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, 1);
    }
};