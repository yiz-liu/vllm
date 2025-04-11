import time
import os
import threading
from queue import deque

def when_exit():
    global timer
    infer_count = len(timer.infer_type_list)
    prefill_token_count = sum([timer.batch_size[i] for i in range(infer_count) if timer.infer_type_list[i]])
    decode_token_count = sum([timer.batch_size[i] for i in range(infer_count) if not timer.infer_type_list[i]])
    max_batch_size = max(timer.batch_size)
    print(f"max_batch_size: {max_batch_size}, prefill_token_count: {prefill_token_count}, decode_token_count: {decode_token_count}")
    offset_dict = {}
    avg_latency_dict = {}
    for key, value in timer.sum_dict.items():
        offset = max(len(value) - infer_count, 0)
        offset_dict[key] = offset
        cur_count = len(value) - offset
        prefill_total_time = sum([value[i + offset] for i in range(cur_count) if timer.infer_type_list[i]])
        decode_total_time = sum([value[i + offset] for i in range(cur_count) if not timer.infer_type_list[i]])
        prefill_token_total_time = sum([value[i + offset] * timer.batch_size[i] for i in range(cur_count) if timer.infer_type_list[i]])
        decode_token_total_time = sum([value[i + offset] * timer.batch_size[i] for i in range(cur_count) if not timer.infer_type_list[i]])
        prefill_wait_time = deque([0] * max_batch_size)
        avg_latency_dict[key] = []
        for i in range(cur_count):
            cur_time = value[i + offset]
            cur_batch_size = timer.batch_size[i]
            cur_total = cur_time * cur_batch_size
            if timer.infer_type_list[i]:
                for j in range(cur_batch_size):
                    if len(prefill_wait_time) > 0:
                        cur_total += prefill_wait_time.popleft()
                prefill_wait_time = deque([k + cur_time for k in prefill_wait_time])
                cur_avg = cur_total / cur_batch_size
                avg_latency_dict[key].append(cur_avg)
            else:
                temp_batch_size = cur_batch_size
                prev_pos = 1
                while i - prev_pos >= 0 and timer.infer_type_list[i - prev_pos]:
                    temp_batch_size -= timer.batch_size[i - prev_pos]
                    cur_total += temp_batch_size * value[i + offset - prev_pos]
                    prev_pos += 1
                cur_avg = cur_total / cur_batch_size
                avg_latency_dict[key].append(cur_avg)
                if cur_batch_size < max_batch_size:
                    for j in range(max_batch_size - cur_batch_size):
                        if j < len(prefill_wait_time):
                            prefill_wait_time[j] += cur_time
                        else:
                            prefill_wait_time.append(cur_time)

        prefill_token_total_latency = sum([avg_latency_dict[key][i] * timer.batch_size[i] for i in range(cur_count) if timer.infer_type_list[i]])
        prefill_token_average_latency = prefill_token_total_latency / prefill_token_count if prefill_token_count > 0 else 0
        prefill_count = timer.infer_type_list[:cur_count].count(True)
        prefill_token_average_time = prefill_token_total_time / prefill_token_count if prefill_token_count > 0 else 0

        decode_token_total_latency = sum([avg_latency_dict[key][i] * timer.batch_size[i] for i in range(cur_count) if not timer.infer_type_list[i]])
        decode_token_average_latency = decode_token_total_latency / decode_token_count if decode_token_count > 0 else 0
        decode_count = timer.infer_type_list[:cur_count].count(False)
        decode_token_average_time = decode_token_total_time / decode_token_count if decode_token_count > 0 else 0
        print(f"{key}:")
        print(f"    prefill count: {prefill_count}, total time: {prefill_total_time}ms, weighted average time: {prefill_token_average_time}ms, weighted average latency: {prefill_token_average_latency}ms")
        print(f"    decode count: {decode_count}, total time: {decode_total_time}ms, weighted average time: {decode_token_average_time}ms, weighted average latency: {decode_token_average_latency}ms")

    out_file = open(f"output_{os.getpid()}.csv", 'w')
    out_file.write(f"count,infer type,batch size,")
    keys = list(timer.sum_dict.keys())
    for key in keys:
        out_file.write(f"{key},{key} avg latency,")
    out_file.write("total,total avg latency\n")
    for i in range(infer_count):
        infer_type = "prefill" if timer.infer_type_list[i] else "decode"
        out_file.write(f"{i},{infer_type},{timer.batch_size[i]},")
        total = 0
        total_avg_latency = 0
        for key in keys:
            cur = 0
            if i + offset_dict[key] < len(timer.sum_dict[key]):
                cur = timer.sum_dict[key][i + offset_dict[key]]
            cur_avg_latency = avg_latency_dict[key][i] if i < len(avg_latency_dict[key]) else 0
            out_file.write(f"{cur},{cur_avg_latency},")
            total += cur
            total_avg_latency += cur_avg_latency
        out_file.write(f"{total},{total_avg_latency}\n")

    out_file.close()
    timer.reset()


class Timer:
    def __init__(self):
        self.time_dict = {}
        self.sum_dict = {}
        self.infer_type_list = [] # True: prompt, False: decode
        self.batch_size = []
        self.task = None

    def start(self, key_str):
        self.time_dict[key_str] = time.time()

    def end(self, key_str, need_print=True):
        if key_str not in self.time_dict:
            return
        end_time = time.time()
        cur_time = (end_time - self.time_dict[key_str]) * 1000
        if need_print:
            print(f"{key_str} time: {cur_time}ms")
        if key_str not in self.sum_dict:
            self.sum_dict[key_str] = [cur_time]
        else:
            self.sum_dict[key_str].append(cur_time)

    def set_infer_info(self, is_prompt, batch_size):
        if self.task is not None:
            self.task.cancel()
        self.task = threading.Timer(10, when_exit, args=())
        self.task.start()
        self.infer_type_list.append(is_prompt)
        self.batch_size.append(batch_size)

    def reset(self):
        self.time_dict = {}
        self.sum_dict = {}
        self.infer_type_list = [] # True: prompt, False: decode
        self.batch_size = []
        self.task = None


timer = Timer()