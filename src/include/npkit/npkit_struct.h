#ifndef NPKIT_STRUCT_H_
#define NPKIT_STRUCT_H_

#include <cstdint>

#pragma pack(push, 1)

union NpKitEvent {
  uint64_t bits[2];
  struct {
    uint64_t type : 8;
    uint64_t size : 32;
    uint64_t rsvd : 24;
    uint64_t timestamp;
  } fields;
};

struct NpKitEventCollectContext {
  NpKitEvent* event_buffer;
  uint64_t event_buffer_head;
};

#pragma pack(pop)

#if defined(ENABLE_NPKIT)

#define NPKIT_GPU_COMM_DECL_FIELDS \
  NpKitEventCollectContext* npKitEventCollectContexts; \
  uint64_t* npKitCpuTimestamp;

#else

#define NPKIT_GPU_COMM_DECL_FIELDS

#endif

#if defined(ENABLE_NPKIT_CPU_EVENTS)

#define NPKIT_CPU_PROXY_DECL_FIELDS \
  int npKitSizesFifo[NCCL_STEPS];

#else

#define NPKIT_CPU_PROXY_DECL_FIELDS

#endif

#endif
