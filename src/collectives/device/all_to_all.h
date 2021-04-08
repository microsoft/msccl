/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "sckl_interpreter.h"

template<int ALGO, int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncAllToAll, ALGO, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      SCKLFunctions<PROTO, FUNC, T, UNROLL> scklfunc;
      scklfunc.run(args);
    }
};
