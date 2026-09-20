# GLM-5.2 Ascend LI C8 简要设计

## 目标

只实现 GLM-5.2 Lightning Indexer 的 INT8 cache：

```text
当前：BF16 index key cache
目标：INT8 index key cache + FP16 scale cache
```

SFA 主 K/V cache 暂时保持 BF16，不处理 SFA C8、PD 分离和 DSA-CP。

模型量化描述中已有：

```text
indexer_quant_type = INT8_DYNAMIC
```

实际需要处理的是带独立 indexer 的层，例如：

```text
6, 10, 14, ..., 66
```

`shared` indexer 层继续复用前一层 top-k，不单独分配 cache。

## 参考实现

参考 DeepSeek-V4 Ascend 的 INT8 LI cache：

```text
python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py
python/sglang/srt/hardware_backend/npu/dsv4/dsv4_memory_pool.py
```

它的核心数据流是：

```python
key_int8, key_scale = torch_npu.npu_dynamic_quant(key)
key_scale = key_scale.to(torch.float16)

pool.write(slot, key_int8, key_scale)
topk = lightning_indexer_quant(query, key_int8_cache, key_scale_cache)
```

GLM-5.2 不直接复用 DeepSeek-V4 的 C4 pool，只复用“INT8 data 与 scale
一起分配、写入、移动和读取”的设计。

## 修改 1：增加 LI C8 开关和层判断

位置：

```text
python/sglang/srt/server_args.py
python/sglang/srt/layers/quantization/modelslim/modelslim.py
```

增加开关，默认关闭：

```text
--enable-sparse-li-c8
```

启用条件：

```python
enable_li_c8 = (
    server_args.enable_sparse_li_c8
    and quant_description.get("indexer_quant_type") == "INT8_DYNAMIC"
)
```

同时解析逐层：

```text
model.layers.N.self_attn.indexer.quant_type
```

只给值为 `INT8_DYNAMIC` 的 full indexer 层启用。启动时输出实际启用的
layer IDs，避免模型名带 `c8` 就自动打开。

## 修改 2：NPU cache pool 增加 INT8 data 和 scale

位置：

```text
python/sglang/srt/hardware_backend/npu/memory_pool_npu.py
```

修改 `NPUMLATokenToKVPool` 的 indexer cache。

当前：

```text
index_k_buffer: BF16 [layer, page, page_size, 1, 128]
```

开启 LI C8 后：

```text
index_k_buffer:       INT8   [layer, page, page_size, 1, 128]
index_k_scale_buffer: FP16   [layer, page, page_size, 1]
```

新增或扩展接口：

```python
get_index_k_buffer(layer_id)
get_index_k_scale_buffer(layer_id)
set_index_k_scale_buffer(layer_id, loc, key, scale)
```

`set_index_k_scale_buffer()` 对 key 和 scale 使用相同的 `loc`：

```python
torch_npu.npu_scatter_nd_update_(key_cache, loc, key_int8)
torch_npu.npu_scatter_nd_update_(scale_cache, loc, key_scale)
```

以下操作也必须同时处理 key 和 scale：

```text
move_kv_cache
get_contiguous_buf_infos
CPU offload/restore（如果当前部署会使用）
get_kv_size_bytes
```

第一版可以不做 packed 单 tensor，先分成两个 tensor，便于验证。

## 修改 3：写入 INT8 LI cache

位置：

```text
python/sglang/srt/layers/attention/dsa/dsa_indexer.py
```

当前 `Indexer.forward_npu()` 生成 BF16 key 后直接：

```python
pool.set_index_k_buffer(layer_id, out_cache_loc, key)
```

改为：

```python
if enable_li_c8_for_this_layer:
    key_int8, key_scale = torch_npu.npu_dynamic_quant(key)
    key_scale = key_scale.to(torch.float16)
    pool.set_index_k_scale_buffer(
        layer_id,
        forward_batch.out_cache_loc,
        key_int8,
        key_scale,
    )
else:
    pool.set_index_k_buffer(
        layer_id,
        forward_batch.out_cache_loc,
        key,
    )
```

需要保证：

- prefill 和 decode 使用同一套 slot mapping；
- `key_int8` 为 contiguous；
- scale 的 token 数与 key 一致；
- shared indexer 层不重复写 cache。

## 修改 4：使用量化 LightningIndexer

位置：

```text
python/sglang/srt/layers/attention/dsa/dsa_indexer.py
```

当前 NPU 路径调用：

```python
torch_npu.npu_lightning_indexer(
    query=query,
    key=bf16_key_cache,
    ...
)
```

LI C8 路径需要参考 vLLM Ascend，调用支持量化输入的算子：

```python
torch.ops._C_ascend.npu_lightning_indexer_quant(
    query=query_int8,
    key=int8_key_cache,
    weights=weights.to(torch.float16),
    query_dequant_scale=query_scale,
    key_dequant_scale=key_scale_cache,
    query_quant_mode=0,
    key_quant_mode=0,
    actual_seq_lengths_query=actual_seq_lengths_query,
    actual_seq_lengths_key=actual_seq_lengths_key,
    block_table=block_table,
    layout_query="TND",
    layout_key="PA_BSND",
    sparse_count=self.index_topk,
    sparse_mode=3,
)
```

query 同样需要 INT8：

```python
query_int8, query_scale = torch_npu.npu_dynamic_quant(query)
query_scale = query_scale.to(torch.float16)
```

如果当前 CANN/torch_npu 镜像没有
`npu_lightning_indexer_quant`，应在启动阶段明确报“不支持 LI C8”，不要
静默把 INT8 cache 传给普通 `npu_lightning_indexer`。

## 修改 5：cache 容量计算

位置：

```text
python/sglang/srt/model_executor/pool_configurator.py
```

当前 NPU index cache 按 BF16 计算：

```text
128 × 2 bytes
```

LI C8 改为：

```text
128 × 1 byte + 1 × 2 bytes scale
```

容量计算必须与实际 pool shape 一致，否则会高估或低估可分配 token 数。

## 最小验证

第一阶段只做三项：

1. 相同 prompt 下比较 BF16 LI 与 C8 LI 的 top-k overlap。
2. 比较最终 logits 和确定性输出，SFA cache 始终保持 BF16。
3. 确认 index cache HBM 从约 `256 bytes/token/layer` 降到约
   `130 bytes/token/layer`。

建议先关闭：

```text
MTP
DSA-CP
prefix cache
PD separation
```

基础路径正确后再逐项恢复。

## 改动范围总结

预计主要修改 4 个位置：

| 文件 | 改动 |
|---|---|
| `server_args.py` / `modelslim.py` | 开关和 eligible layer 判断 |
| `memory_pool_npu.py` | INT8 key cache、FP16 scale cache 及生命周期 |
| `dsa_indexer.py` | key/query 动态量化和量化 LightningIndexer |
| `pool_configurator.py` | LI C8 容量计算 |

这不是从零实现：cache data/scale 生命周期参考 DeepSeek-V4，量化
LightningIndexer 的调用参数参考 vLLM Ascend。第一版不碰 SFA 主 cache，
改动量应控制在中等范围。
