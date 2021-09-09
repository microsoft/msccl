#include <chrono>
#include <fstream>
#include <unistd.h>

#include "alloc.h"
#include "npkit/npkit.h"

uint64_t NpKit::rank_ = 0;

NpKitEvent* NpKit::gpu_event_buffers_[MAXCHANNELS] = {0};
NpKitEvent* NpKit::gpu_event_caches_[MAXCHANNELS] = {0};
NpKitEvent* NpKit::cpu_event_buffers_[MAXCHANNELS] = {0};

NpKitEventCollectContext* NpKit::gpu_collect_contexts_[MAXCHANNELS] = {0};
NpKitEventCollectContext* NpKit::cpu_collect_contexts_[MAXCHANNELS] = {0};
uint64_t* NpKit::cpu_timestamp_ = nullptr;

std::thread* NpKit::cpu_timestamp_update_thread_ = nullptr;
volatile bool NpKit::cpu_timestamp_update_thread_should_stop_ = false;

void NpKit::CpuTimestampUpdateThread() {
  uint64_t init_system_clock = std::chrono::system_clock::now().time_since_epoch().count();
  uint64_t init_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
  uint64_t curr_steady_clock = 0;
  volatile uint64_t* volatile_cpu_timestamp_ = cpu_timestamp_;
  while (!cpu_timestamp_update_thread_should_stop_) {
    curr_steady_clock = std::chrono::steady_clock::now().time_since_epoch().count();
    *volatile_cpu_timestamp_ = init_system_clock + (curr_steady_clock - init_steady_clock);
  }
}

ncclResult_t NpKit::Init(int rank) {
  uint64_t i = 0;
  NpKitEventCollectContext context;
  context.event_buffer_head = 0;
  rank_ = rank;

  for (i = 0; i < MAXCHANNELS; i++) {
    // Init event buffers
    NCCLCHECK(ncclCudaCalloc(&(gpu_event_buffers_[i]), kMaxNumEventsPerChannel));
    NCCLCHECK(ncclCudaCalloc(&(gpu_event_caches_[i]), kMaxNumEventsCachedPerChannel));
    NCCLCHECK(ncclCalloc(&(cpu_event_buffers_[i]), kMaxNumEventsPerChannel));

    // Init collect contexts
    context.event_buffer = gpu_event_buffers_[i];
    context.event_cache = gpu_event_caches_[i];
    NCCLCHECK(ncclCudaCalloc(&(gpu_collect_contexts_[i]), 1));
    NCCLCHECK(ncclCudaMemcpy(gpu_collect_contexts_[i], &context, 1));

    context.event_buffer = cpu_event_buffers_[i];
    context.event_cache = nullptr;
    NCCLCHECK(ncclCalloc(&(cpu_collect_contexts_[i]), 1));
    NCCLCHECK(ncclCudaMemcpy(cpu_collect_contexts_[i], &context, 1));
  }

  // Init timestamp
  NCCLCHECK(ncclCudaHostCalloc(&cpu_timestamp_, 1));
  cpu_timestamp_update_thread_should_stop_ = false;
  cpu_timestamp_update_thread_ = new std::thread(CpuTimestampUpdateThread);

  return ncclSuccess;
}

ncclResult_t NpKit::Dump(const std::string& dump_dir) {
  uint64_t i = 0;
  std::string dump_file_path;

  // Dump CPU events
  for (i = 0; i < MAXCHANNELS; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/cpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    auto cpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    cpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[i]),
        cpu_collect_contexts_[i]->event_buffer_head * sizeof(NpKitEvent));
    cpu_trace_file.close();
  }

  // Dump CPU clock info
  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_num_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_num_str = std::to_string(std::chrono::steady_clock::duration::period::num);
  auto clock_period_num_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_num_file.write(clock_period_num_str.c_str(), clock_period_num_str.length());
  clock_period_num_file.close();

  dump_file_path = dump_dir;
  dump_file_path += "/cpu_clock_period_den_rank_";
  dump_file_path += std::to_string(rank_);
  std::string clock_period_den_str = std::to_string(std::chrono::steady_clock::duration::period::den);
  auto clock_period_den_file = std::fstream(dump_file_path, std::ios::out);
  clock_period_den_file.write(clock_period_den_str.c_str(), clock_period_den_str.length());
  clock_period_den_file.close();

  // Dump GPU events, reuse CPU struct
  for (i = 0; i < MAXCHANNELS; i++) {
    dump_file_path = dump_dir;
    dump_file_path += "/gpu_events_rank_";
    dump_file_path += std::to_string(rank_);
    dump_file_path += "_channel_";
    dump_file_path += std::to_string(i);
    NCCLCHECK(ncclCudaMemcpy(cpu_event_buffers_[i], gpu_event_buffers_[i], kMaxNumEventsPerChannel));
    NCCLCHECK(ncclCudaMemcpy(cpu_collect_contexts_[i], gpu_collect_contexts_[i], 1));
    auto gpu_trace_file = std::fstream(dump_file_path, std::ios::out | std::ios::binary);
    gpu_trace_file.write(reinterpret_cast<char*>(cpu_event_buffers_[i]),
        cpu_collect_contexts_[i]->event_buffer_head * sizeof(NpKitEvent));
    gpu_trace_file.close();
  }

  // Dump GPU clockRate
  dump_file_path = dump_dir;
  dump_file_path += "/gpu_clock_rate_rank_";
  dump_file_path += std::to_string(rank_);
  cudaDeviceProp dev_prop;
  int dev;
  CUDACHECK(cudaGetDevice(&dev));
  CUDACHECK(cudaGetDeviceProperties(&dev_prop, dev));
  std::string clock_rate_str = std::to_string(dev_prop.clockRate);
  auto gpu_clock_rate_file = std::fstream(dump_file_path, std::ios::out);
  gpu_clock_rate_file.write(clock_rate_str.c_str(), clock_rate_str.length());
  gpu_clock_rate_file.close();

  return ncclSuccess;
}

ncclResult_t NpKit::Shutdown() {
  uint64_t i = 0;

  // Stop CPU timestamp updating thread
  cpu_timestamp_update_thread_should_stop_ = true;
  cpu_timestamp_update_thread_->join();

  for (i = 0; i < MAXCHANNELS; i++) {
    // Free event buffers
    CUDACHECK(cudaFree(gpu_event_buffers_[i]));
    CUDACHECK(cudaFree(gpu_event_caches_[i]));
    free(cpu_event_buffers_[i]);

    // Free collect contexts
    CUDACHECK(cudaFree(gpu_collect_contexts_[i]));
    free(cpu_collect_contexts_[i]);
  }

  // Free timestamp
  NCCLCHECK(ncclCudaHostFree(cpu_timestamp_));

  return ncclSuccess;
}

NpKitEventCollectContext* NpKit::GetGpuEventCollectContext(int channel) {
  return gpu_collect_contexts_[channel];
}

NpKitEventCollectContext* NpKit::GetCpuEventCollectContext(int channel) {
  return cpu_collect_contexts_[channel];
}

void NpKit::GenerateCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                             NpKitEvent* event) {
  event->fields.type = type;
  event->fields.size = size;
  event->fields.rsvd = rsvd;
  event->fields.timestamp = timestamp;
}

void NpKit::CollectCpuEvent(const NpKitEvent& event, NpKitEventCollectContext* collect_context) {
  uint64_t event_buffer_head = collect_context->event_buffer_head;
  NpKitEvent* event_buffer = collect_context->event_buffer;
  if (event_buffer_head < kMaxNumEventsPerChannel) {
    event_buffer[event_buffer_head] = event;
    collect_context->event_buffer_head++;
  }
}

void NpKit::GenerateAndCollectCpuEvent(uint8_t type, uint32_t size, uint32_t rsvd, uint64_t timestamp,
                                       NpKitEvent* event, NpKitEventCollectContext* collect_context) {
  GenerateCpuEvent(type, size, rsvd, timestamp, event);
  CollectCpuEvent(*event, collect_context);
}

uint64_t* NpKit::GetCpuTimeStamp() {
  return cpu_timestamp_;
}
