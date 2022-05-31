#include "devcomm.h"

static int strideMemcpyGridsize = 0, strideMemcpyBlocksize = 0;

// memory stride copy kernel
template <typename T>
__global__ void strideMemcpyKernel(T *__restrict__ out, const T *__restrict__ in, const size_t size, const int height, const int width) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = tid; i < size * height * width; i += gridDim.x * blockDim.x) {
    const size_t index = i / size, offset = i % size;
    const size_t j = (width * (index % height) + (index / height)) * size + offset;
    out[j] = in[i];
  }
}

cudaError_t strideMemcpyAsync(void *dst, const void *src, const size_t size, const int height, const int width, cudaStream_t stream) {
  if (strideMemcpyGridsize == 0 || strideMemcpyBlocksize == 0)
    cudaOccupancyMaxPotentialBlockSize(&strideMemcpyGridsize, &strideMemcpyBlocksize, strideMemcpyKernel<uint4>);

  if (size < sizeof(uint4))
    strideMemcpyKernel<char><<<strideMemcpyGridsize, strideMemcpyBlocksize, 0, stream>>>((char*)dst, (char*)src, size, height, width);
  else
    strideMemcpyKernel<uint4><<<strideMemcpyGridsize, strideMemcpyBlocksize, 0, stream>>>((uint4*)dst, (uint4*)src, size/sizeof(uint4), height, width);
  return cudaSuccess;
}
