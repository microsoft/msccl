// NOTE: Doesn't support LL protocol

#include "comm.h"
#include "shm.h"

#define NCCL_SHMCPU_MAX_REQUESTS 8
static_assert(NCCL_STEPS <= NCCL_SHMCPU_MAX_REQUESTS, "Not enough SHMCPU ring buffer space to cover for steps");

#define MAX_SHM_NAME_LEN 64
#define MAX_CONNECTORS 64

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
} *g_params = nullptr;

uint64_t g_num_connectors = 0;

static void shmOpenNoCuda(const char* shmname, const int shmsize, void** shmPtr, int create) {
  int fd = -1;
  void* ptr = MAP_FAILED;
  shmSetup(shmname, shmsize, &fd, &ptr, create);
  *shmPtr = ptr;
}

static void shmCloseNoCuda(void* shmPtr, const int shmsize) {
  munmap(shmPtr, shmsize);
}

struct shmCpuConnectInfo {
  uint64_t pidHash;
  int id;
  int sendRank;
  int recvRank;
};

struct shmCpuSendResources {
  struct ncclSendMem* selfCpuSendMem; // Peer CPU updates head as consumer
  struct ncclRecvMem* peerCpuRecvMem; // Self CPU updates tail as producer
  char* selfCpuBuffer;  // local-socket D2H
  uint64_t cpuCurrTail;
  int sendSizes[NCCL_SHMCPU_MAX_REQUESTS];

  struct ncclSendMem* gpuSendMem; // CPU updates head as consumer
  struct ncclRecvMem* gpuRecvMem; // GPU updates tail as producer
  char* gpuBuffer;
  uint64_t gpuStep;
  cudaStream_t gpuD2HStream;
  cudaEvent_t gpuD2HEvents[NCCL_SHMCPU_MAX_REQUESTS];

  int buffSize;
  int buffOffsets[NCCL_NUM_PROTOCOLS];

  int connectorId;
  uint64_t cudaProxyTail[NCCL_SHMCPU_MAX_REQUESTS];
};

struct shmCpuRecvResources {
  struct ncclSendMem* peerCpuSendMem; // Self CPU updates head as consumer
  struct ncclRecvMem* selfCpuRecvMem; // Peer CPU updates tail as producer
  char* peerCpuBuffer;  // cross-socket H2D
  uint64_t cpuCurrHead;

  struct ncclSendMem* gpuSendMem; // GPU updates head as consumer
  struct ncclRecvMem* gpuRecvMem; // CPU updates tail as producer
  char* gpuBuffer;
  uint64_t gpuStep;
  cudaStream_t gpuH2DStream;
  cudaEvent_t gpuH2DEvents[NCCL_SHMCPU_MAX_REQUESTS];

  int buffSize;
  int buffOffsets[NCCL_NUM_PROTOCOLS];

  int connectorId;
  uint64_t cudaProxyTail[NCCL_SHMCPU_MAX_REQUESTS];
};

/* Determine two peers can communicate with SHMCPU */
ncclResult_t shmCpuCanConnect(int* ret, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo* info1, struct ncclPeerInfo* info2) {
  *ret = 0;

  // Same host?
  if (info1->hostHash != info2->hostHash) return ncclSuccess;

  // Common /dev/shm (between containers) ?
  if (info1->shmDev != info2->shmDev) return ncclSuccess;

  *ret = 1;

  return ncclSuccess;
}

#define MAX_SHMCPU_NAME_LEN 1024

/* Determine if we will use this transport for this peer and return connect
 * information for this peer */
ncclResult_t shmCpuSendSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* send, int channelId, int connIndex) {
  struct shmCpuSendResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  send->transportResources = resources;

  // Allocate proxy FIFOs for CPU-GPU communication
  send->conn.shared = 0;
  send->proxyAppendPtr = &send->proxyAppend;
  NCCLCHECK(ncclCudaHostCalloc(&resources->gpuSendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->gpuRecvMem, 1));
  send->conn.tail = &resources->gpuRecvMem->tail;
  send->conn.sizesFifo = resources->gpuRecvMem->sizesFifo;
  send->conn.ptrsFifo = NULL;
  send->conn.head = &resources->gpuSendMem->head;
  for (int i=0; i<NCCL_STEPS; i++) send->conn.sizesFifo[i] = -1;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffOffsets[p] = resources->buffSize;
    resources->buffSize += send->comm->buffSizes[p];
  }
  NCCLCHECK(ncclCudaCalloc(&(resources->gpuBuffer), resources->buffSize));
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    send->conn.buffs[p] = resources->gpuBuffer + offset;
    offset += send->comm->buffSizes[p];
  }

  char shmCpuName[MAX_SHMCPU_NAME_LEN];

  // Connect to CUDA proxy
  if (g_params == nullptr) {
    sprintf(shmCpuName, "cuda-proxy-params-%d", myInfo->rank);
    shmOpenNoCuda(shmCpuName, sizeof(struct CudaProxyControlParams), (void**)(&g_params), 0);
    NCCLCHECK(shmUnlink(shmCpuName));
  }
  // set device
  int cuda_device;
  ncclCommCuDevice(comm, &cuda_device);
  volatile int *device = &(g_params->device);
  *device = cuda_device;
  cudaSetDevice(cuda_device);
  // set gpu buffer
  CUDACHECK(cudaIpcGetMemHandle(&(g_params->gpu_buffer_handle[g_num_connectors]), resources->gpuBuffer));

  // Allocate SHM buffer for IPC
  struct shmCpuConnectInfo info;
  void* devPtr;
  info.id = channelId;
  info.pidHash = myInfo->pidHash;
  info.sendRank = myInfo->rank;
  info.recvRank = peerInfo->rank;

  sprintf(shmCpuName, "nccl-shmcpu-send-head-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  NCCLCHECK(shmOpen(shmCpuName, sizeof(struct ncclSendMem), (void**)(&resources->selfCpuSendMem), &devPtr, 1));
  sprintf(shmCpuName, "nccl-shmcpu-send-buffer-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  NCCLCHECK(shmOpen(shmCpuName, resources->buffSize, (void**)(&resources->selfCpuBuffer), &devPtr, 1));

  // set cpu_buffer_handle
  strcpy(g_params->cpu_buffer_handle[g_num_connectors], shmCpuName);
  volatile uint64_t* buffer_size = &(g_params->buffer_size[g_num_connectors]);
  *buffer_size = resources->buffSize;
  resources->connectorId = g_num_connectors;
  // set num_connectors
  g_num_connectors++;
  volatile uint64_t* num_connectors = &(g_params->num_connectors);
  *num_connectors = g_num_connectors;

  INFO(NCCL_INIT|NCCL_SHMCPU,"Channel %02d : %d[%lx] -> %d[%lx] via SHMCPU", channelId, myInfo->rank, myInfo->busId, peerInfo->rank, peerInfo->busId);
  static_assert(sizeof(struct shmCpuConnectInfo) <= sizeof(struct ncclConnect), "shmcpu Connect Recv Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmCpuConnectInfo));

  // Create GPU stream
  //CUDACHECK(cudaStreamCreateWithFlags(&(resources->gpuD2HStream), cudaStreamNonBlocking));
  //for (int i=0; i<NCCL_STEPS; i++) cudaEventCreateWithFlags(resources->gpuD2HEvents + i, cudaEventDisableTiming);

  return ncclSuccess;
}

ncclResult_t shmCpuRecvSetup(struct ncclComm* comm, struct ncclTopoGraph* graph, struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connectInfo, struct ncclConnector* recv, int channelId, int connIndex) {
  struct shmCpuRecvResources* resources;
  NCCLCHECK(ncclCalloc(&resources, 1));
  recv->transportResources = resources;

  // Allocate proxy FIFOs for CPU-GPU communication
  recv->conn.shared = 0;
  recv->proxyAppendPtr = &recv->proxyAppend;
  NCCLCHECK(ncclCudaHostCalloc(&resources->gpuSendMem, 1));
  NCCLCHECK(ncclCudaHostCalloc(&resources->gpuRecvMem, 1));
  recv->conn.tail = &resources->gpuRecvMem->tail;
  recv->conn.ptrsFifo = NULL;
  recv->conn.head = &resources->gpuSendMem->head;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    resources->buffOffsets[p] = resources->buffSize;
    resources->buffSize += recv->comm->buffSizes[p];
  }
  NCCLCHECK(ncclCudaCalloc(&(resources->gpuBuffer), resources->buffSize));
  int offset = 0;
  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    recv->conn.buffs[p] = resources->gpuBuffer + offset;
    offset += recv->comm->buffSizes[p];
  }

  // Allocate SHM buffer for IPC
  struct shmCpuConnectInfo info;
  char shmCpuName[MAX_SHMCPU_NAME_LEN];
  void* devPtr;
  info.id = channelId;
  info.pidHash = myInfo->pidHash;
  info.sendRank = peerInfo->rank;
  info.recvRank = myInfo->rank;

  sprintf(shmCpuName, "nccl-shmcpu-recv-tail-%lx-%d-%d-%d", info.pidHash, info.id, info.sendRank, info.recvRank);
  NCCLCHECK(shmOpen(shmCpuName, sizeof(struct ncclRecvMem), (void**)(&resources->selfCpuRecvMem), &devPtr, 1));
  for (int i=0; i<NCCL_SHMCPU_MAX_REQUESTS; i++) resources->selfCpuRecvMem->sizesFifo[i] = -1;

  INFO(NCCL_INIT|NCCL_SHMCPU,"Channel %02d : %d[%lx] -> %d[%lx] via SHMCPU", channelId, peerInfo->rank, peerInfo->busId, myInfo->rank, myInfo->busId);
  static_assert(sizeof(struct shmCpuConnectInfo) <= sizeof(struct ncclConnect), "shmcpu Connect Send Info is too big");
  memcpy(connectInfo, &info, sizeof(struct shmCpuConnectInfo));

  // Create GPU stream
  //CUDACHECK(cudaStreamCreateWithFlags(&(resources->gpuH2DStream), cudaStreamNonBlocking));
  //for (int i=0; i<NCCL_STEPS; i++) cudaEventCreateWithFlags(resources->gpuH2DEvents + i, cudaEventDisableTiming);

  return ncclSuccess;
}

ncclResult_t shmCpuSendConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* send) {
  // Setup device pointers
  struct shmCpuSendResources* resources = (struct shmCpuSendResources*)send->transportResources;
  struct shmCpuConnectInfo* info = (struct shmCpuConnectInfo*)connectInfo;
  void* devPtr;

  char shmCpuName[MAX_SHMCPU_NAME_LEN];
  sprintf(shmCpuName, "nccl-shmcpu-recv-tail-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  NCCLCHECK(shmOpen(shmCpuName, sizeof(struct ncclRecvMem), (void**)(&resources->peerCpuRecvMem), &devPtr, 0));
  NCCLCHECK(shmUnlink(shmCpuName));

  return ncclSuccess;
}

/* Connect to this peer */
ncclResult_t shmCpuRecvConnect(struct ncclComm* comm, struct ncclConnect* connectInfo, int nranks, int rank, struct ncclConnector* recv) {
  // Setup device pointers
  struct shmCpuRecvResources* resources = (struct shmCpuRecvResources*)recv->transportResources;
  struct shmCpuConnectInfo* info = (struct shmCpuConnectInfo*)connectInfo;
  void* devPtr;

  char shmCpuName[MAX_SHMCPU_NAME_LEN];

  // Connect to CUDA proxy
  if (g_params == nullptr) {
    sprintf(shmCpuName, "cuda-proxy-params-%d", rank);
    shmOpenNoCuda(shmCpuName, sizeof(struct CudaProxyControlParams), (void**)(&g_params), 0);
    NCCLCHECK(shmUnlink(shmCpuName));
  }
  // set device
  int cuda_device;
  ncclCommCuDevice(comm, &cuda_device);
  volatile int *device = &(g_params->device);
  *device = cuda_device;
  cudaSetDevice(cuda_device);
  // set gpu buffer
  CUDACHECK(cudaIpcGetMemHandle(&(g_params->gpu_buffer_handle[g_num_connectors]), resources->gpuBuffer));

  sprintf(shmCpuName, "nccl-shmcpu-send-head-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  NCCLCHECK(shmOpen(shmCpuName, sizeof(struct ncclSendMem), (void**)(&resources->peerCpuSendMem), &devPtr, 0));
  NCCLCHECK(shmUnlink(shmCpuName));
  sprintf(shmCpuName, "nccl-shmcpu-send-buffer-%lx-%d-%d-%d", info->pidHash, info->id, info->sendRank, info->recvRank);
  NCCLCHECK(shmOpen(shmCpuName, resources->buffSize, (void**)(&resources->peerCpuBuffer), &devPtr, 0));

  // set cpu_buffer_handle
  strcpy(g_params->cpu_buffer_handle[g_num_connectors], shmCpuName);
  volatile uint64_t* buffer_size = &(g_params->buffer_size[g_num_connectors]);
  *buffer_size = resources->buffSize;
  resources->connectorId = g_num_connectors;
  // set num_connectors
  g_num_connectors++;
  volatile uint64_t* num_connectors = &(g_params->num_connectors);
  *num_connectors = g_num_connectors;

  return ncclSuccess;
}

ncclResult_t shmCpuSendFree(void* transportResources) {
  struct shmCpuSendResources* resources = (struct shmCpuSendResources*)transportResources;

  NCCLCHECK(ncclCudaHostFree(resources->gpuSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->gpuRecvMem));
  CUDACHECK(cudaFree(resources->gpuBuffer));

  NCCLCHECK(shmClose(resources->selfCpuSendMem, NULL, sizeof(struct ncclSendMem)));
  NCCLCHECK(shmClose(resources->selfCpuBuffer, NULL, resources->buffSize));

  free(resources);

  volatile uint64_t* should_stop = &(g_params->should_stop);
  *should_stop = 1;

  return ncclSuccess;
}

ncclResult_t shmCpuRecvFree(void* transportResources) {
  struct shmCpuRecvResources* resources = (struct shmCpuRecvResources*)transportResources;

  NCCLCHECK(ncclCudaHostFree(resources->gpuSendMem));
  NCCLCHECK(ncclCudaHostFree(resources->gpuRecvMem));
  CUDACHECK(cudaFree(resources->gpuBuffer));

  NCCLCHECK(shmClose(resources->selfCpuRecvMem, NULL, sizeof(struct ncclRecvMem)));

  free(resources);

  volatile uint64_t* should_stop = &(g_params->should_stop);
  *should_stop = 1;

  return ncclSuccess;
}

ncclResult_t shmCpuSendProxy(struct ncclProxyArgs* args) {
  volatile uint64_t *params_ready = &(g_params->params_ready);
  *params_ready = 1;
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmCpuSendResources* resources = (struct shmCpuSendResources*) (sub->connector->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->gpuStep, args->chunkSteps);
      sub->posted = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (sub->done == sub->nsteps) continue;
      struct shmCpuSendResources* resources = (struct shmCpuSendResources*) (sub->connector->transportResources);
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      int buffSize = stepSize*args->sliceSteps;
      if (sub->sendbytes < buffSize) buffSize = sub->sendbytes;
      // Post buffers to the GPU
      if (sub->posted < sub->nsteps && sub->posted < sub->done + NCCL_STEPS) {
        sub->posted += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      // Check whether we received data from the GPU and send it to the ring buffer
      if (sub->transmitted < sub->posted && sub->transmitted < sub->done + NCCL_STEPS) {
        int buffSlot = (sub->base+sub->transmitted)%NCCL_STEPS;
        volatile int* sizesFifo = resources->gpuRecvMem->sizesFifo;
        volatile uint64_t* recvTail = &resources->gpuRecvMem->tail;
        volatile uint64_t* cpuBuffHead = &resources->selfCpuSendMem->head;
        if (sizesFifo[buffSlot] != -1 && *recvTail > sub->base+sub->transmitted && resources->cpuCurrTail < *cpuBuffHead + NCCL_STEPS) {
          // Data and buffer is ready, try to send.
          int size = sizesFifo[buffSlot];
          char* buff = localBuff+buffSlot*stepSize;
          char* cpuBuff = resources->selfCpuBuffer + resources->buffOffsets[p] + buffSlot*stepSize;

	  /*
          CUDACHECK(cudaMemcpyAsync(cpuBuff, buff, size, cudaMemcpyDeviceToHost, resources->gpuD2HStream));
          CUDACHECK(cudaEventRecord(resources->gpuD2HEvents[buffSlot], resources->gpuD2HStream));
	  */
	  uint64_t tail = *(volatile uint64_t*)(&(g_params->d2h_tail));
          *(volatile uint64_t*)(&(resources->cudaProxyTail[buffSlot])) = tail;
	  tail %= NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS;
          *(volatile uint64_t*)(&(g_params->d2h_connector_id[tail])) = resources->connectorId;
          *(volatile uint64_t*)(&(g_params->d2h_cpu_buff_offset[tail])) = cpuBuff - resources->selfCpuBuffer;
          *(volatile uint64_t*)(&(g_params->d2h_gpu_buff_offset[tail])) = buff - resources->gpuBuffer;
          *(volatile uint64_t*)(&(g_params->d2h_size[tail])) = size;
          (*(volatile uint64_t*)(&(g_params->d2h_tail)))++;
          resources->sendSizes[buffSlot] = size;

          sizesFifo[buffSlot] = -1;
          // Make sure size is reset to zero before we update the head.
          __sync_synchronize();
          sub->transmitted += args->sliceSteps;
          resources->cpuCurrTail += args->sliceSteps;
          args->idle = 0;
          continue;
        }
      }
      // Check whether ring buffer has completed some send operations.
      if (sub->done < sub->transmitted) {
        int done = 0;
        int buffSlot = (sub->base+sub->done)%NCCL_STEPS;

	/*
        cudaError_t cuda_err = cudaEventQuery(resources->gpuD2HEvents[buffSlot]);
        if (cuda_err == cudaSuccess) done = 1;
        else if (cuda_err != cudaErrorNotReady) CUDACHECK(cuda_err);
	*/
        uint64_t tail = *(volatile uint64_t*)(&(resources->cudaProxyTail[buffSlot]));
        if (*(volatile uint64_t*)(&(g_params->d2h_head)) > tail) done = 1;
	//done = 1;

        if (done) {
          sub->done += args->sliceSteps;
          resources->gpuSendMem->head = sub->base + sub->done;
          // update CPU ring buffer's tail and size
          resources->peerCpuRecvMem->sizesFifo[buffSlot] = resources->sendSizes[buffSlot];
          resources->peerCpuRecvMem->tail += args->sliceSteps;
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            resources->gpuStep = sub->base + sub->nsteps;
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

ncclResult_t shmCpuRecvProxy(struct ncclProxyArgs* args) {
  volatile uint64_t *params_ready = &(g_params->params_ready);
  *params_ready = 1;
  if (args->state == ncclProxyOpReady) {
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      struct shmCpuRecvResources* resources = (struct shmCpuRecvResources*) (sub->connector->transportResources);
      // Round to next multiple of sliceSteps
      sub->base = ROUNDUP(resources->gpuStep, args->chunkSteps);
      sub->posted = sub->received = sub->transmitted = sub->done = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  args->idle = 1;
  if (args->state == ncclProxyOpProgress) {
    int p = args->protocol;
    for (int s=0; s<args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs+s;
      if (sub->done == sub->nsteps) continue;
      struct shmCpuRecvResources* resources = (struct shmCpuRecvResources*) (sub->connector->transportResources);
      int stepSize = sub->connector->comm->buffSizes[p] / NCCL_STEPS;
      char* localBuff = sub->connector->conn.buffs[p];
      int buffSize = stepSize*args->sliceSteps;
      if (sub->recvbytes < buffSize) buffSize = sub->recvbytes;
      volatile int* cpuSizesFifo = resources->selfCpuRecvMem->sizesFifo;
      volatile uint64_t* cpuBuffTail = &resources->selfCpuRecvMem->tail;
      int buffSlot = (sub->base+sub->posted)%NCCL_STEPS;
      if (sub->posted < sub->done + NCCL_STEPS && sub->posted < sub->nsteps && cpuSizesFifo[buffSlot] != -1 && resources->cpuCurrHead + args->sliceSteps <= *cpuBuffTail) {
        char* ptr = localBuff+buffSlot*stepSize;
        char* cpuBuff = resources->peerCpuBuffer + resources->buffOffsets[p] + buffSlot*stepSize;
        int size = cpuSizesFifo[buffSlot];
        cpuSizesFifo[buffSlot] = -1;

	/*
        CUDACHECK(cudaMemcpyAsync(ptr, cpuBuff, size, cudaMemcpyHostToDevice, resources->gpuH2DStream));
        CUDACHECK(cudaEventRecord(resources->gpuH2DEvents[buffSlot], resources->gpuH2DStream));
	*/
	uint64_t tail = *(volatile uint64_t*)(&(g_params->h2d_tail));
        *(volatile uint64_t*)(&(resources->cudaProxyTail[buffSlot])) = tail;
	tail %= NCCL_SHMCPU_MAX_REQUESTS*MAX_CONNECTORS;
        *(volatile uint64_t*)(&(g_params->h2d_connector_id[tail])) = resources->connectorId;
        *(volatile uint64_t*)(&(g_params->h2d_cpu_buff_offset[tail])) = cpuBuff - resources->peerCpuBuffer;
        *(volatile uint64_t*)(&(g_params->h2d_gpu_buff_offset[tail])) = ptr - resources->gpuBuffer;
        *(volatile uint64_t*)(&(g_params->h2d_size[tail])) = size;
        (*(volatile uint64_t*)(&(g_params->h2d_tail)))++;

        sub->posted += args->sliceSteps;
        resources->cpuCurrHead += args->sliceSteps;
        args->idle = 0;
        continue;
      }
      if (sub->posted > sub->received) {
        int buffSlot = (sub->base+sub->received)%NCCL_STEPS;
        int done = 0;

	/*
        cudaError_t cuda_err = cudaEventQuery(resources->gpuH2DEvents[buffSlot]);
        if (cuda_err == cudaSuccess) done = 1;
        else if (cuda_err != cudaErrorNotReady) CUDACHECK(cuda_err);
	*/
        uint64_t tail = *(volatile uint64_t*)(&(resources->cudaProxyTail[buffSlot]));
        if (*(volatile uint64_t*)(&(g_params->h2d_head)) > tail) done = 1;
	//done = 1;

        if (done) {
          sub->received += args->sliceSteps;
          sub->transmitted += args->sliceSteps;
          __sync_synchronize();
          resources->peerCpuSendMem->head += args->sliceSteps;
          resources->gpuRecvMem->tail = sub->base + sub->transmitted;
          args->idle = 0;
          continue;
        }
      }
      if (sub->transmitted > sub->done) {
        volatile uint64_t* sendHead = &resources->gpuSendMem->head;
        uint64_t done = *sendHead;
        while (done > sub->base + sub->done &&
            // LL and LL128 can acknowledge 0-bytes send before they even happen. Don't go past what we transmitted.
            sub->transmitted > sub->done) {
          sub->done += args->sliceSteps;
          args->idle = 0;
          if (sub->done == sub->nsteps) {
            resources->gpuStep = sub->base + sub->nsteps;
            args->done++;
          }
        }
      }
    }
    if (args->done == args->nsubs) {
      args->state = ncclProxyOpNone;
    }
  }
  return ncclSuccess;
}

struct ncclTransport shmCpuTransport = {
  "SHMCPU",
  shmCpuCanConnect,
  { shmCpuSendSetup, shmCpuSendConnect, shmCpuSendFree, shmCpuSendProxy },
  { shmCpuRecvSetup, shmCpuRecvConnect, shmCpuRecvFree, shmCpuRecvProxy }
};
