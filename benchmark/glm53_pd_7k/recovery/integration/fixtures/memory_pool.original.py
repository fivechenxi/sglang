# Extracted unchanged pre-fix methods, for negative controls only.
class MLATokenToKVPool:

    def get_cpu_copy(self, indices, mamba_indices=None):
        current_platform.synchronize()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i:i + chunk_size]
                kv_cpu = self.kv_buffer[layer_id][chunk_indices].to('cpu', non_blocking=True)
                kv_cache_cpu[-1].append(kv_cpu)
        current_platform.synchronize()
        return kv_cache_cpu
