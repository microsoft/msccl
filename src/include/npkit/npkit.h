#ifndef NPKIT_H_
#define NPKIT_H_

#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "devcomm.h"
#include "npkit/npkit_event.h"
#include "npkit/npkit_struct.h"

class NpKit {
 public:
  static const uint64_t kMaxNumEventsCachedPerChannel = 128;

  static ncclResult_t Init(int rank);

  static ncclResult_t Dump(const std::string& dump_dir);

  static ncclResult_t Shutdown();

  static NpKitEventCollectContext* GetGpuEventCollectContext(int channel);

  static inline __device__ void GenerateGpuEvent(uint8_t type, uint32_t size,
                                                 uint32_t rsvd, uint64_t timestamp,
                                                 NpKitEvent* event) {
    event->fields.type = type;
    event->fields.size = size;
    event->fields.rsvd = rsvd;
    event->fields.timestamp = timestamp;
  }

  static inline __device__ void CollectGpuEvent(const NpKitEvent& event,
                                                NpKitEventCollectContext* collect_context) {
    uint64_t event_buffer_head = collect_context->event_buffer_head;
    NpKitEvent* event_buffer = collect_context->event_buffer;
    if (event_buffer_head < kMaxNumEventsPerChannel) {
      event_buffer[event_buffer_head] = event;
      collect_context->event_buffer_head++;
    }
  }

  static inline __device__ void GenerateAndCollectGpuEvent(uint8_t type, uint32_t size,
                                                           uint32_t rsvd, uint64_t timestamp,
                                                           NpKitEvent* event,
                                                           NpKitEventCollectContext* collect_context) {
    GenerateGpuEvent(type, size, rsvd, timestamp, event);
    CollectGpuEvent(*event, collect_context);
  }

  static NpKitEventCollectContext* GetCpuEventCollectContext(int channel);

  static void GenerateCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                               NpKitEvent* event);


  static void CollectCpuEvent(const NpKitEvent& event,
                              NpKitEventCollectContext* collect_context);

  static void GenerateAndCollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                                        NpKitEvent* event, NpKitEventCollectContext* collect_context);

  static uint64_t* GetCpuTimeStamp();

 private:
  static void CpuTimestampUpdateThread();

  static const uint64_t kMaxNumEventsPerChannel = 1ULL << 20;

  static NpKitEvent* gpu_event_buffers_[MAXCHANNELS];
  static NpKitEvent* gpu_event_caches_[MAXCHANNELS];
  static NpKitEvent* cpu_event_buffers_[MAXCHANNELS];

  static NpKitEventCollectContext* gpu_collect_contexts_[MAXCHANNELS];
  static NpKitEventCollectContext* cpu_collect_contexts_[MAXCHANNELS];
  static uint64_t* cpu_timestamp_;

  static uint64_t rank_;

  static std::thread* cpu_timestamp_update_thread_;
  static volatile bool cpu_timestamp_update_thread_should_stop_;
};

#endif