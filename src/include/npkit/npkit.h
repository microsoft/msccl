#ifndef NPKIT_H_
#define NPKIT_H_

#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "npkit/npkit_event.h"
#include "npkit/npkit_struct.h"

class NpKit {
 public:
  static const uint64_t kNumGpuEventBuffers = 512;

  static const uint64_t kNumCpuEventBuffers = 32;

  static ncclResult_t Init(int rank);

  static ncclResult_t Dump(const std::string& dump_dir);

  static ncclResult_t Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContexts();

  static inline __device__ void CollectGpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                                                NpKitEventCollectContext* ctx) {
    uint64_t event_buffer_head = ctx->event_buffer_head;
    if (event_buffer_head < kMaxNumGpuEventsPerBuffer) {
      NpKitEvent& event = ctx->event_buffer[event_buffer_head];
      event.fields.type = type;
      event.fields.size = size;
      event.fields.rsvd = rsvd;
      event.fields.timestamp = timestamp;
      ctx->event_buffer_head++;
    }
  }

  static void CollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp, int channel_id);

  static uint64_t* GetCpuTimestamp();

 private:
  static void CpuTimestampUpdateThread();

  // 64K * 512 * 16B = 512MB per GPU
  static const uint64_t kMaxNumGpuEventsPerBuffer = 1ULL << 16;

  // 64K * 2 (send/recv) * (512/32) = 2M, 2M * 32 * 16B = 1GB per CPU
  static const uint64_t kMaxNumCpuEventsPerBuffer = 1ULL << 21;

  static NpKitEvent** gpu_event_buffers_;
  static NpKitEvent** cpu_event_buffers_;

  static NpKitEventCollectContext* gpu_collect_contexts_;
  static NpKitEventCollectContext* cpu_collect_contexts_;
  static uint64_t* cpu_timestamp_;

  static uint64_t rank_;

  static std::thread* cpu_timestamp_update_thread_;
  static volatile bool cpu_timestamp_update_thread_should_stop_;
};

#if defined(ENABLE_NPKIT)

#define NPKIT_GPU_SET_CTX_ID(__prims__) \
  if (__thread_flag__) { \
    __prims__.__ctx_id__ = __ctx_id__; \
  }

#define NPKIT_GPU_SYNC_TIME_SHARED() \
  if (__thread_flag__) { \
    NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *(ncclShmem.comm.npKitCpuTimestamp), \
        ncclShmem.comm.npKitEventCollectContexts + __ctx_id__); \
    NpKit::CollectGpuEvent(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, clock64(), \
        ncclShmem.comm.npKitEventCollectContexts + __ctx_id__); \
  }

#define NPKIT_GPU_SYNC_TIME(__bid__, __tid__) \
  int __ctx_id__ = __bid__; \
  bool __thread_flag__ = (__tid__ == 0); \
  NPKIT_GPU_SYNC_TIME_SHARED()

#define NPKIT_GPU_SYNC_TIME_TREE_SPLIT(__bid__, __tid__, __nthreadsSplit__) \
  bool __thread_flag__ = false; \
  int __ctx_id__ = 0; \
  if (__tid__ == 0) { \
    __thread_flag__ = true; \
    __ctx_id__ = __bid__ * 2; \
  } else if (tree->up != -1 && __tid__ == __nthreadsSplit__) { \
    __thread_flag__ = true; \
    __ctx_id__ = __bid__ * 2 + 1; \
  } \
  NPKIT_GPU_SYNC_TIME_SHARED()

#define NPKIT_GPU_SYNC_TIME_SEND(__bid__, __tid__) \
  int __ctx_id__ = __bid__ * NCCL_MAX_WORK_ELEMENTS_P2P; \
  bool __thread_flag__ = (__tid__ == 0); \
  NPKIT_GPU_SYNC_TIME_SHARED()

#define NPKIT_GPU_SYNC_TIME_RECV(__bid__, __tid__) \
  int __ctx_id__ = __bid__ * NCCL_MAX_WORK_ELEMENTS_P2P + 1; \
  bool __thread_flag__ = (__tid__ == 0); \
  NPKIT_GPU_SYNC_TIME_SHARED()

#define NPKIT_GPU_ENTER_EVENT(__type__, __size__) \
  if (tid == 0) { \
    NpKit::CollectGpuEvent(__type__, __size__, 0, clock64(), \
        ncclShmem.comm.npKitEventCollectContexts + __ctx_id__); \
  }

#define NPKIT_GPU_COLLECT_EVENT(__type__, __size__) \
  if (tid == 0) { \
    NpKit::CollectGpuEvent(__type__, __size__, __npKitWaitTotalTime__, clock64(), \
        ncclShmem.comm.npKitEventCollectContexts + __ctx_id__); \
  }

#define NPKIT_GPU_PRIMS_DECL_FIELDS \
  public: \
    int __ctx_id__ = 0; \
  private: \
    uint64_t __npKitWaitEntryTime__ = 0; \
    uint64_t __npKitWaitExitTime__ = 0; \
    uint64_t __npKitWaitTotalTime__ = 0;

#define NPKIT_GPU_PRIMS_OP_INIT(__tid__) \
  if (__tid__ == 0) { \
    __npKitWaitTotalTime__ = 0; \
  }

#define NPKIT_GPU_PRIMS_WAIT_BEGIN(__tid__) \
  if (__tid__ == 0) { \
    __npKitWaitEntryTime__ = clock64(); \
  }

#define NPKIT_GPU_PRIMS_WAIT_END(__tid__) \
  if (__tid__ == 0) { \
    __npKitWaitExitTime__ = clock64(); \
    __npKitWaitTotalTime__ += __npKitWaitExitTime__ - __npKitWaitEntryTime__; \
  }

#define NPKIT_GPU_PRIMS_WAIT_BEGIN_WITH_SPIN(__tid__) \
  int npKitWaitSpins = 0; \
  if (__tid__ == 0) { \
    __npKitWaitEntryTime__ = clock64(); \
  }

#define NPKIT_GPU_PRIMS_WAIT_INC_SPIN() \
  npKitWaitSpins++;

#define NPKIT_GPU_PRIMS_WAIT_END_WITH_SPIN(__tid__) \
  if (__tid__ == 0) { \
    __npKitWaitExitTime__ = clock64(); \
    __npKitWaitTotalTime__ += (__npKitWaitExitTime__ - __npKitWaitEntryTime__) * (npKitWaitSpins - 1) / npKitWaitSpins; \
  }

#else

#define NPKIT_GPU_SET_CTX_ID(__prims__)

#define NPKIT_GPU_SYNC_TIME_TREE_SPLIT(__bid__, __tid__, __nthreadsSplit__)

#define NPKIT_GPU_SYNC_TIME_SEND(__bid__, __tid__)

#define NPKIT_GPU_SYNC_TIME_RECV(__bid__, __tid__)

#define NPKIT_GPU_SYNC_TIME(__bid__, __tid__)

#define NPKIT_GPU_ENTER_EVENT(__type__, __size__)

#define NPKIT_GPU_COLLECT_EVENT(__type__, __size__)

#define NPKIT_GPU_PRIMS_DECL_FIELDS

#define NPKIT_GPU_PRIMS_OP_INIT(__tid__)

#define NPKIT_GPU_PRIMS_WAIT_BEGIN(__tid__)

#define NPKIT_GPU_PRIMS_WAIT_END(__tid__)

#define NPKIT_GPU_PRIMS_WAIT_BEGIN_WITH_SPIN(__tid__)

#define NPKIT_GPU_PRIMS_WAIT_INC_SPIN()

#define NPKIT_GPU_PRIMS_WAIT_END_WITH_SPIN(__tid__)

#endif

#if defined(ENABLE_NPKIT)

#define NPKIT_CPU_COLLECT_EVENT(__ctx_id__, __type__, __size__, __rsvd__) \
  NpKit::CollectCpuEvent(__type__, __size__, __rsvd__, \
      *(volatile uint64_t*)NpKit::GetCpuTimestamp(), __ctx_id__); \

#define NPKIT_CPU_PROXY_SAVE_SIZE() \
  sub->npKitSizesFifo[buffSlot] = size;

#else

#define NPKIT_CPU_COLLECT_EVENT(__ctx_id__, __type__, __size__, __rsvd__)

#define NPKIT_CPU_PROXY_SAVE_SIZE()

#endif

#if defined(ENABLE_NPKIT)

#define NPKIT_INIT() \
  NCCLCHECK(NpKit::Init(comm->rank)); \
  comm->hostDevComm.npKitEventCollectContexts = NpKit::GetGpuEventCollectContexts(); \
  comm->hostDevComm.npKitCpuTimestamp = NpKit::GetCpuTimestamp();

#define NPKIT_TEARDOWN() \
  const char* npKitDumpDir = getenv("NPKIT_DUMP_DIR"); \
  if (npKitDumpDir == nullptr) { \
    npKitDumpDir = "/tmp/"; \
  } \
  NCCLCHECK(NpKit::Dump(npKitDumpDir)); \
  NCCLCHECK(NpKit::Shutdown());

#else

#define NPKIT_INIT()

#define NPKIT_TEARDOWN()

#endif

#endif
