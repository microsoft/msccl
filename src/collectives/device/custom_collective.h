/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "msccl_interpreter.h"
 
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_MSCCL, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionSimple<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, 1);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_MSCCL, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL128<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, 1);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_MSCCL, NCCL_PROTO_LL, FUNC, T, UNROLL> {
    public:
    __device__ void run(struct ncclWorkElem* args) {
      mscclFunctionLL<FUNC, T, UNROLL> mscclfunc;
      mscclfunc.run(args, 1);
    }
};

//FIXME: Find a way to remove below declarations for RING, TREE, and COLLNET.
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
    }
};
