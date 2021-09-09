import argparse
import os
import json

from queue import Queue

def parse_npkit_event_header(npkit_event_header_path):
    npkit_event_def = {'id_to_type': {}, 'type_to_id': {}}
    with open(npkit_event_header_path, 'r') as f:
        lines = [x.strip() for x in f.readlines() if len(x.strip()) != 0]
        line_idx = 0
        while line_idx < len(lines):
            if lines[line_idx].startswith('#define NPKIT_EVENT_'):
                fields = lines[line_idx].split()
                if len(fields) == 3:
                    event_type = fields[1]
                    event_id = int(fields[2], 0)
                    npkit_event_def['type_to_id'][event_type] = event_id
                    npkit_event_def['id_to_type'][event_id] = event_type
            line_idx += 1
    return npkit_event_def

def parse_gpu_clock_scale(gpu_clock_file_path):
    with open(gpu_clock_file_path, 'r') as f:
        freq_in_khz = f.read()
        return float(freq_in_khz) * 1e3 / 1e6

def parse_cpu_clock_scale(cpu_clock_den_file_path, cpu_clock_num_file_path):
    with open(cpu_clock_num_file_path, 'r') as f:
        num = float(f.read())
    with open(cpu_clock_den_file_path, 'r') as f:
        den = float(f.read())
    return den / num / 1e6

def parse_gpu_event(event_bytes):
    return {
        'id': int.from_bytes(event_bytes[0:1], byteorder='little', signed=False),
        'size': int.from_bytes(event_bytes[1:5], byteorder='little', signed=False),
        'rsvd': int.from_bytes(event_bytes[5:8], byteorder='little', signed=False),
        'timestamp': int.from_bytes(event_bytes[8:16], byteorder='little', signed=False)
    }

def parse_cpu_event(event_bytes):
    return {
        'id': int.from_bytes(event_bytes[0:1], byteorder='little', signed=False),
        'size': int.from_bytes(event_bytes[1:5], byteorder='little', signed=False),
        'slot': int.from_bytes(event_bytes[5:8], byteorder='little', signed=False),
        'timestamp': int.from_bytes(event_bytes[8:16], byteorder='little', signed=False)
    }

def parse_gpu_event_file(npkit_dump_dir, npkit_event_def, rank, channel, gpu_clock_scale, cpu_clock_scale, num_kernel_runs, num_kernel_runs_to_sample):
    gpu_event_file_path = os.path.join(npkit_dump_dir, 'gpu_events_rank_%d_channel_%d' % (rank, channel))
    raw_event_size = 16
    curr_cpu_base_time = None
    curr_gpu_base_time = None
    gpu_events = []
    event_type_to_seq = {}
    with open(gpu_event_file_path, 'rb') as f:
        raw_content = f.read()
        raw_content = raw_content[-len(raw_content) * num_kernel_runs_to_sample // num_kernel_runs:]
        raw_content_size = len(raw_content)
        raw_content_idx = 0
        gpu_stage_durations = []
        while raw_content_idx < raw_content_size:
            parsed_gpu_event = parse_gpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
            if npkit_event_def['id_to_type'][parsed_gpu_event['id']] == 'NPKIT_EVENT_TIME_SYNC_CPU':
                curr_cpu_base_time = parsed_gpu_event['timestamp'] / cpu_clock_scale
                curr_gpu_base_time = None
            elif npkit_event_def['id_to_type'][parsed_gpu_event['id']] == 'NPKIT_EVENT_TIME_SYNC_GPU':
                if curr_gpu_base_time is None:
                    curr_gpu_base_time = parsed_gpu_event['timestamp'] / gpu_clock_scale
            else:
                if curr_gpu_base_time is None:
                    curr_gpu_base_time = parsed_gpu_event['timestamp'] / gpu_clock_scale
                event_type = npkit_event_def['id_to_type'][parsed_gpu_event['id']]
                phase = 'B' if event_type.endswith('_ENTRY') else 'E'
                gpu_events.append({
                    'ph': phase,
                    'ts': curr_cpu_base_time + parsed_gpu_event['timestamp'] / gpu_clock_scale - curr_gpu_base_time,
                    'pid': rank,
                    'tid': channel
                })
                if phase == 'B':
                    if event_type not in event_type_to_seq:
                        event_type_to_seq[event_type] = 0
                    gpu_events[-1].update({
                        'name': event_type,
                        'cat': 'GPU',
                        'args': {
                            'rank': rank,
                            'channel': channel,
                            'seq': event_type_to_seq[event_type]
                        }
                    })
                    event_type_to_seq[event_type] += 1
                else:
                    gpu_events[-1]['args'] = {'size': parsed_gpu_event['size'], 'rsvd': parsed_gpu_event['rsvd']}
                    delta_time = gpu_events[-1]['ts'] - gpu_events[-2]['ts']
                    gpu_events[-1]['args']['bw (GB/s)'] = gpu_events[-1]['args']['size'] / delta_time / 1e3
                    gpu_stage_durations.append(delta_time)
            raw_content_idx += raw_event_size
        num_stages = len(gpu_stage_durations) // num_kernel_runs_to_sample
        gpu_stage_durations_avg = [0.] * num_stages
        for idx, duration in enumerate(gpu_stage_durations):
            gpu_stage_durations_avg[idx % num_stages] += duration / num_kernel_runs_to_sample
    return gpu_events, gpu_stage_durations_avg

def parse_cpu_event_file(npkit_dump_dir, npkit_event_def, rank, channel, cpu_clock_scale, num_kernel_runs, num_kernel_runs_to_sample):
    cpu_event_file_path = os.path.join(npkit_dump_dir, 'cpu_events_rank_%d_channel_%d' % (rank, channel))
    raw_event_size = 16
    cpu_events = []
    event_type_to_seq = {}

    fiber_is_usable = []
    fiber_open_ts = []
    slot_to_fiber_id = {}
    channel_shift = 100

    with open(cpu_event_file_path, 'rb') as f:
        raw_content = f.read()
        raw_content = raw_content[-len(raw_content) * num_kernel_runs_to_sample // num_kernel_runs:]
        raw_content_size = len(raw_content)
        raw_content_idx = 0
        cpu_stage_durations = []
        while raw_content_idx < raw_content_size:
            parsed_cpu_event = parse_cpu_event(raw_content[raw_content_idx : raw_content_idx + raw_event_size])
            event_type = npkit_event_def['id_to_type'][parsed_cpu_event['id']]
            phase = 'B' if event_type.endswith('_POSTED') else 'E'
            cpu_events.append({
                'ph': phase,
                'ts': parsed_cpu_event['timestamp'] / cpu_clock_scale,
                'pid': rank
            })
            slot = parsed_cpu_event['slot']
            if phase == 'B':
                # Open fiber event
                fiber_id = 0
                while fiber_id < len(fiber_is_usable):
                    if fiber_is_usable[fiber_id]:
                        break
                    fiber_id += 1
                if fiber_id == len(fiber_is_usable):
                    fiber_is_usable.append(True)
                    fiber_open_ts.append(0.0)
                slot_to_fiber_id[slot] = fiber_id
                fiber_open_ts[fiber_id] = cpu_events[-1]['ts']
                fiber_is_usable[fiber_id] = False

                if event_type not in event_type_to_seq:
                    event_type_to_seq[event_type] = 0
                cpu_events[-1].update({
                    'name': event_type,
                    'cat': 'CPU',
                    'args': {
                        'rank': rank,
                        'channel': channel,
                        'slot': parsed_cpu_event['slot'],
                        'seq': event_type_to_seq[event_type]
                    }
                })
                event_type_to_seq[event_type] += 1
            else:
                # Close fiber event
                fiber_id = slot_to_fiber_id[slot]
                slot_to_fiber_id.pop(slot)
                last_ts = fiber_open_ts[fiber_id]
                fiber_is_usable[fiber_id] = True

                delta_time = max(0.001, cpu_events[-1]['ts'] - last_ts)
                cpu_events[-1]['args'] = {'size': parsed_cpu_event['size']}
                cpu_events[-1]['args']['bw (GB/s)'] = \
                cpu_events[-1]['args']['size'] / delta_time / 1e3
                cpu_stage_durations.append(delta_time)

            cpu_events[-1]['tid'] = fiber_id + (channel + 1) * channel_shift

            raw_content_idx += raw_event_size
        num_stages = len(cpu_stage_durations) // num_kernel_runs_to_sample
        cpu_stage_durations_avg = [0.] * num_stages
        for idx, duration in enumerate(cpu_stage_durations):
            cpu_stage_durations_avg[idx % num_stages] += duration / num_kernel_runs_to_sample
    return cpu_events, cpu_stage_durations_avg

def convert_npkit_dump_to_trace(npkit_dump_dir, output_dir, npkit_event_def, num_kernel_runs, num_kernel_runs_to_sample):
    files_in_dump_dir = next(os.walk(npkit_dump_dir))[2]
    gpu_event_files = [x for x in files_in_dump_dir if x.startswith('gpu_events_rank_')]

    ranks = list(set([int(x.split('_rank_')[1].split('_')[0]) for x in gpu_event_files]))
    channels = list(set([int(x.split('_channel_')[1].split('_')[0]) for x in gpu_event_files]))

    trace = {'traceEvents': []}
    gpu_stage_durations_channels = {}
    cpu_stage_durations_channels = {}

    for rank in ranks:
        cpu_clock_den_file_path = os.path.join(npkit_dump_dir, 'cpu_clock_period_den_rank_%d' % rank)
        cpu_clock_num_file_path = os.path.join(npkit_dump_dir, 'cpu_clock_period_num_rank_%d' % rank)
        cpu_clock_scale = parse_cpu_clock_scale(cpu_clock_den_file_path, cpu_clock_num_file_path)

        gpu_clock_file_path = os.path.join(npkit_dump_dir, 'gpu_clock_rate_rank_%d' % rank)
        gpu_clock_scale = parse_gpu_clock_scale(gpu_clock_file_path)

        gpu_stage_durations_channels[rank] = {}
        cpu_stage_durations_channels[rank] = {}

        for channel in channels:
            gpu_events, gpu_stage_durations = parse_gpu_event_file(npkit_dump_dir, npkit_event_def, rank, channel, gpu_clock_scale, cpu_clock_scale, num_kernel_runs, num_kernel_runs_to_sample)
            trace['traceEvents'].extend(gpu_events)
            gpu_stage_durations_channels[rank][channel] = gpu_stage_durations
            cpu_events, cpu_stage_durations = parse_cpu_event_file(npkit_dump_dir, npkit_event_def, rank, channel, cpu_clock_scale, num_kernel_runs, num_kernel_runs_to_sample)
            trace['traceEvents'].extend(cpu_events)
            cpu_stage_durations_channels[rank][channel] = cpu_stage_durations

    trace['traceEvents'].sort(key=lambda x : x['ts'])
    trace['displayTimeUnit'] = 'ns'

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'npkit_event_trace.json'), 'w') as f:
        json.dump(trace, f)
    with open(os.path.join(output_dir, 'gpu_stage_durations.json'), 'w') as f:
        json.dump(gpu_stage_durations_channels, f)
    with open(os.path.join(output_dir, 'cpu_stage_durations.json'), 'w') as f:
        json.dump(cpu_stage_durations_channels, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--npkit_dump_dir', type=str, required=True, help='NPKit dump directory.')
    parser.add_argument('--npkit_event_header_path', type=str, required=True, help='Path to npkit_event.h.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory.')
    parser.add_argument('--num_kernel_runs', type=int, required=True, help='Number of kernel runs.')
    parser.add_argument('--num_kernel_runs_to_sample', type=int, required=True, help='Last number of kernel runs to sample.')
    args = parser.parse_args()

    npkit_event_def = parse_npkit_event_header(args.npkit_event_header_path)
    convert_npkit_dump_to_trace(args.npkit_dump_dir, args.output_dir, npkit_event_def, args.num_kernel_runs, args.num_kernel_runs_to_sample)
