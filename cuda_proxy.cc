#include <chrono>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>

#include "shm.h"

#define NCCL_SHMCPU_MAX_REQUESTS 8
#define MAX_SHM_NAME_LEN 64
#define MAX_CONNECTORS 64

int g_rank = 0;

struct CudaProxyControlParams {
  uint64_t queue_depth;

  cudaIpcMemHandle_t gpu_buffer_handle[MAX_CONNECTORS];
  char cpu_buffer_handle[MAX_CONNECTORS][MAX_SHM_NAME_LEN];
  uint64_t buffer_size[MAX_CONNECTORS];
  int device;
  uint64_t params_ready;
  uint64_t num_connectors;

  char* gpu_buffer[MAX_CONNECTORS];
  char* cpu_buffer[MAX_CONNECTORS];

  uint64_t d2h_head;
  uint64_t d2h_tail;
  uint64_t d2h_connector_id[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t d2h_gpu_buff_offset[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t d2h_cpu_buff_offset[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t d2h_size[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];

  uint64_t h2d_head;
  uint64_t h2d_tail;
  uint64_t h2d_connector_id[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t h2d_gpu_buff_offset[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t h2d_cpu_buff_offset[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t h2d_size[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];

  uint64_t should_stop;
};

struct CudaProxyStatus {
  cudaStream_t d2h_stream;
  cudaEvent_t d2h_events[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  std::chrono::time_point<std::chrono::steady_clock> d2h_starts[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t d2h_submitted;

  cudaStream_t h2d_stream;
  cudaEvent_t h2d_events[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  std::chrono::time_point<std::chrono::steady_clock> h2d_starts[NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS];
  uint64_t h2d_submitted;
} g_status;

struct CudaProxyControlParams* g_params;

void OpenCudaProxyControlParams() {
  char shm_name[MAX_SHM_NAME_LEN];
  sprintf(shm_name, "cuda-proxy-params-%d", g_rank);
  shmOpen(shm_name, sizeof(struct CudaProxyControlParams), (void**)(&g_params), 1, 0);
}

void CloseCudaProxyControlParams() {
  shmClose(g_params, sizeof(struct CudaProxyControlParams), 0);
}

void Run() {
  printf("Waiting connection\n");
  volatile uint64_t* params_ready = &(g_params->params_ready);
  while (!*params_ready);
  printf("Connection accepted\n");

  cudaError_t err = cudaSuccess;
  err = cudaSetDevice(g_rank);
  if (err != cudaSuccess) printf("cudaError in cudaSetDevice: %lu\n", (uint64_t)err);

  cudaStreamCreateWithFlags(&(g_status.d2h_stream), cudaStreamNonBlocking);
  for (int i=0; i<NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS; i++) cudaEventCreateWithFlags(&(g_status.d2h_events[i]), cudaEventDisableTiming);
  g_status.d2h_submitted = 0;
  cudaStreamCreateWithFlags(&(g_status.h2d_stream), cudaStreamNonBlocking);
  for (int i=0; i<NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS; i++) cudaEventCreateWithFlags(&(g_status.h2d_events[i]), cudaEventDisableTiming);
  g_status.h2d_submitted = 0;

  /*
  char* debugGpuBuffer = nullptr;
  cudaMalloc(&debugGpuBuffer, 4096);
  char* debugCpuBuffer = nullptr;
  cudaMallocHost(&debugCpuBuffer, 4096);

  uint64_t summed_micros = 0;
  g_status.h2d_starts[0] = std::chrono::steady_clock::now();
  for (int i = 0; i < 100; i++) {
    //g_status.h2d_starts[0] = std::chrono::steady_clock::now();
    cudaMemcpy(debugGpuBuffer, debugCpuBuffer, 1024, cudaMemcpyHostToDevice);
    //cudaMemcpyAsync(debugGpuBuffer, debugCpuBuffer, 1024, cudaMemcpyHostToDevice, g_status.h2d_stream);
    //cudaEventRecord(g_status.h2d_events[0], g_status.h2d_stream);
    //while (cudaSuccess != cudaEventQuery(g_status.h2d_events[0]));
    //auto duration = std::chrono::steady_clock::now() - g_status.h2d_starts[0];
    //summed_micros += std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  }
  cudaStreamSynchronize(g_status.h2d_stream);
  auto duration = std::chrono::steady_clock::now() - g_status.h2d_starts[0];
  summed_micros += std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
  printf("avg micros: %lu\n", summed_micros / 100);
  sleep(100);
  */

  printf("num_connectors: %lu\n", g_params->num_connectors);
  printf("device: %d\n", g_params->device);
  for (int i = 0; i < g_params->num_connectors; i++) {
    printf("cpu_buffer_handle: %s, size: %lu\n", g_params->cpu_buffer_handle[i], g_params->buffer_size[i]);
    shmOpen(g_params->cpu_buffer_handle[i], g_params->buffer_size[i], (void**)(&(g_params->cpu_buffer[i])), 0, 1);
    err = cudaIpcOpenMemHandle((void**)(&(g_params->gpu_buffer[i])), g_params->gpu_buffer_handle[i], cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) printf("cudaError in cudaIpcOpenMemHandle: %lu %d\n", (uint64_t)err, g_params->gpu_buffer_handle[i].reserved[0]);
  }

  volatile uint64_t* should_stop = &(g_params->should_stop);
  while (!*should_stop) {
    volatile uint64_t* d2h_head = &(g_params->d2h_head);
    volatile uint64_t* d2h_tail = &(g_params->d2h_tail);
    if (*d2h_tail > g_status.d2h_submitted) {
      // check new d2h
      uint64_t d2h_idx = g_status.d2h_submitted % (NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS);
      //printf("cudaMemcpyAsyncD2H start %lu\n", d2h_idx);
      uint64_t connector_id = *(volatile uint64_t*)(&(g_params->d2h_connector_id[d2h_idx]));
      uint64_t cpu_offset = *(volatile uint64_t*)(&(g_params->d2h_cpu_buff_offset[d2h_idx]));
      uint64_t gpu_offset = *(volatile uint64_t*)(&(g_params->d2h_gpu_buff_offset[d2h_idx]));
      uint64_t size = *(volatile uint64_t*)(&(g_params->d2h_size[d2h_idx]));
      char* cpu_buff = g_params->cpu_buffer[connector_id];
      char* gpu_buff = g_params->gpu_buffer[connector_id];
      g_status.d2h_starts[d2h_idx] = std::chrono::steady_clock::now();
      //printf("d2h size: %lu\n", size);
      err = cudaMemcpyAsync(cpu_buff + cpu_offset, gpu_buff + gpu_offset, size, cudaMemcpyDeviceToHost, g_status.d2h_stream);
      if (err != cudaSuccess) printf("cudaMemcpyAsync D2H error: %d\n", err);
      cudaEventRecord(g_status.d2h_events[d2h_idx], g_status.d2h_stream);
      if (err != cudaSuccess) printf("cudaEventRecord D2H error: %d\n", err);
      g_status.d2h_submitted++;
    } else if (g_status.d2h_submitted > *d2h_head) {
      // check d2h completion
      uint64_t d2h_idx = (*d2h_head) % (NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS);
      err = cudaEventQuery(g_status.d2h_events[d2h_idx]);
      if (err == cudaSuccess) {
        //printf("cudaMemcpyAsyncD2H stop %lu\n", d2h_idx);
	auto duration = std::chrono::steady_clock::now() - g_status.d2h_starts[d2h_idx];
	uint64_t micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	//printf("cudaMemcpyAsyncD2H duration: %lu\n", micros);
        (*d2h_head)++;
      } else if (err != cudaErrorNotReady) printf("cudaEventQuery D2H error: %d\n", err);
    }

    volatile uint64_t* h2d_head = &(g_params->h2d_head);
    volatile uint64_t* h2d_tail = &(g_params->h2d_tail);
    if (*h2d_tail > g_status.h2d_submitted) {
      // check new h2d
      uint64_t h2d_idx = g_status.h2d_submitted % (NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS);
      //printf("cudaMemcpyAsyncH2D start %lu\n", h2d_idx);
      uint64_t connector_id = *(volatile uint64_t*)(&(g_params->h2d_connector_id[h2d_idx]));
      uint64_t cpu_offset = *(volatile uint64_t*)(&(g_params->h2d_cpu_buff_offset[h2d_idx]));
      uint64_t gpu_offset = *(volatile uint64_t*)(&(g_params->h2d_gpu_buff_offset[h2d_idx]));
      uint64_t size = *(volatile uint64_t*)(&(g_params->h2d_size[h2d_idx]));
      char* cpu_buff = g_params->cpu_buffer[connector_id];
      char* gpu_buff = g_params->gpu_buffer[connector_id];
      g_status.h2d_starts[h2d_idx] = std::chrono::steady_clock::now();
      //printf("h2d size: %lu\n", size);
      err = cudaMemcpyAsync(gpu_buff + gpu_offset, cpu_buff + cpu_offset, size, cudaMemcpyHostToDevice, g_status.h2d_stream);
      //err = cudaMemcpyAsync(debugGpuBuffer, debugCpuBuffer, size, cudaMemcpyHostToDevice, g_status.h2d_stream);
      if (err != cudaSuccess) printf("cudaMemcpyAsync H2D error: %d\n", err);
      err = cudaEventRecord(g_status.h2d_events[h2d_idx], g_status.h2d_stream);
      if (err != cudaSuccess) printf("cudaEventRecord H2D error: %d\n", err);
      g_status.h2d_submitted++;
    } else if (g_status.h2d_submitted > *h2d_head) {
      // check h2d completion
      uint64_t h2d_idx = (*h2d_head) % (NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS);
      err = cudaEventQuery(g_status.h2d_events[h2d_idx]);
      if (err == cudaSuccess) {
        //printf("cudaMemcpyAsyncH2D stop %lu\n", h2d_idx);
	auto duration = std::chrono::steady_clock::now() - g_status.h2d_starts[h2d_idx];
	uint64_t micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	//printf("cudaMemcpyAsyncH2D duration: %lu\n", micros);
        (*h2d_head)++;
      } else if (err != cudaErrorNotReady) printf("cudaEventQuery H2D error: %d\n", err);
    }
  }
  printf("Connection closed\n");

  for (int i = 0; i < g_params->num_connectors; i++) {
    shmUnlink(g_params->cpu_buffer_handle[i]);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  printf("Rank: %d\n", g_rank);
  printf("sizeof(CudaProxyControlParams): %lu\n", sizeof(CudaProxyControlParams));
  OpenCudaProxyControlParams();
  Run();
  CloseCudaProxyControlParams();
  MPI_Finalize();
  return 0;
}
