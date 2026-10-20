# Extracted unchanged pre-fix methods, for negative controls only.
class Req:

    def offload_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[self.req_pool_idx, :self.seqlen - 1]
        self.kv_cache_cpu = token_to_kv_pool_allocator.get_cpu_copy(token_indices, mamba_indices=self.mamba_pool_idx)

    def load_kv_cache(self, req_to_token_pool, token_to_kv_pool_allocator):
        token_indices = req_to_token_pool.req_to_token[self.req_pool_idx, :self.seqlen - 1]
        token_to_kv_pool_allocator.load_cpu_copy(self.kv_cache_cpu, token_indices, mamba_indices=self.mamba_pool_idx)
        del self.kv_cache_cpu
