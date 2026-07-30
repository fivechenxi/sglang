# GLM-5.2 C8 在 Ascend 上的实现与 SGLang 改造

## 1. 先说结论

当前 GLM-5.2 W4A8 已经通过以下两个上游补丁跑通：

- #31707：将 `w13/w2_scale_bias` 从 3D 转成 GroupedMatmul 要求的 2D。
- #32113：修正 W4 权重 transpose、contiguous 和 INT32 packing 顺序。

剩下的 **C8 与 W4A8 MoE 无关**。模型现在能够运行，只能说明 W4A8
计算已经跑通，不能说明 C8 已启用。

C8 的目标是把稀疏注意力在历史 token 上保存的 cache 从 BF16 压成
INT8。GLM-5.2 的稀疏注意力包含两个阶段，因此 vLLM Ascend 把 C8
拆成两个独立能力：

| 能力 | 缓存对象 | 用途 |
|---|---|---|
| LI C8 | Lightning Indexer 的 key/index cache | 从历史 token 中选出 top-k |
| SFA C8 | Sparse Flash Attention 的主 K/V cache | 对选出的 token 计算 attention |

最适合 SGLang 的实施顺序是：

```text
W4A8 + BF16 cache（当前）
        ↓
W4A8 + LI C8 + BF16 SFA cache
        ↓
W4A8 + LI C8 + SFA C8
```

先做 LI C8，再做 SFA C8。两者不要绑成一个开关。

## 2. C8 在计算链里的位置

GLM-5.2 的稀疏 attention 可以简化成：

```text
当前 token
   │
   ├─ 生成 index query
   │
   │      LI index cache（全部历史 token）
   │                │
   └───────────────打分
                    │
                 top-k token
                    │
                    ▼
             SFA 主 K/V cache
                    │
                    ▼
              attention output
```

不开 C8 时，两份 cache 都用 BF16：

```text
LI index cache = BF16
SFA K/V cache  = BF16
```

开启 C8 后，可以分别变成：

```text
LI index cache = INT8 data + scale
SFA K/V cache  = packed INT8 data + scale
```

这里最容易混淆的是：checkpoint 中 indexer 的 `INT8_DYNAMIC` 描述主要
表示 indexer 线性层如何计算，不等于运行时 LI cache 已经自动使用 INT8。
运行时 cache 是否是 C8，取决于 cache allocator、写入算子和读取算子是否
全部选择了 C8 路径。

## 3. INT8 cache 如何工作

以一段 BF16 cache 向量 `x` 为例，写入 cache 前做量化：

```text
q = clamp(round(x / scale), -128, 127)
```

实际保存：

```text
q     : INT8 cache data
scale : FP16/FP32 quantization scale
```

读取时有两种方式：

1. kernel 内部直接消费 `INT8 + scale`；
2. 先反量化：

   ```text
   x ≈ q × scale
   ```

第一种才有性能价值。第二种可以作为正确性 fallback，但每步反量化整段
cache 会抵消显存带宽收益。

BF16 每元素 2 bytes，INT8 每元素 1 byte，因此 cache 数据部分理论上接近
减半。实际收益会略低，因为还要保存 scale、对齐和 metadata。

## 4. LI C8 的实现

LI 的任务是给所有历史 token 打分并选 top-k。它不执行最终 attention，
只负责找出“应该关注哪些 token”。

### 4.1 Cache 结构

不开 LI C8：

```text
indexer_cache = BF16 index key
```

开启 LI C8：

```text
indexer_cache = INT8 index key
indexer_scale = FP16/FP32 scale
```

vLLM Ascend 会为 indexer 单独建立 cache spec。启用 LI C8 时，indexer
cache 从一个 BF16 tensor 变成两个 tensor：

```text
[quantized index key, scale]
```

这说明 LI scale 必须跟 index cache block 使用相同的 slot/block 映射，
不能只分配一块全局 scale 后忽略 prefix cache 和 block reuse。

### 4.2 写入

prefill 或 decode 生成新 token 的 index key 后：

```text
BF16 index key
      ↓ quantize/store operator
INT8 key + scale
      ↓ 写入当前 token 对应 cache slot
```

需要处理：

- token 到 cache slot 的映射；
- prefill 一次写多个 token；
- decode 每步追加一个 token；
- prefix cache 复用已有 block；
- block 释放后 data 和 scale 一起失效。

### 4.3 读取和 top-k

LI kernel 的输入从：

```text
query + BF16 index cache
```

改为：

```text
query + INT8 index cache + scale
```

kernel 在打分过程中完成反量化或量化点积，最后输出 top-k token 索引。
如果只是把 cache 分配成 INT8，但 LI kernel 仍按 BF16 地址解释，就会产生
错误结果；所以 allocator 和 kernel 必须同时改。

### 4.4 为什么先做 LI C8

LI C8 相对独立：

- 不改变主 K/V cache 的 layout；
- 不改变最终 SFA attention 的输入格式；
- 可以单独比较 C8 开关前后的 top-k；
- 出错时可以直接回退到 BF16 index cache。

第一阶段应只保证：

```text
LI top-k(C8) 与 LI top-k(BF16) 高度一致
```

主 attention 仍使用 BF16 cache。

## 5. SFA C8 的实现

SFA 使用 LI 选出的 token 进行最终 sparse attention。它消费的是主 K/V
cache，因此改动范围比 LI C8 大。

### 5.1 Packed cache

不开 SFA C8 时，主 cache 通常有独立的 K/V 或 MLA/DSA 状态 tensor。
vLLM Ascend 的 SFA C8 将主 cache 组织成 packed byte tensor，并在同一
layout 中保存量化后的 attention 状态和必要的量化数据。

可以抽象为：

```text
BF16:
  main_cache = [K/state tensor, V/state tensor]

C8:
  main_cache = [packed INT8 tensor]
```

具体 packed layout 必须由 SFA kernel 的接口决定，不能简单调用
`tensor.to(torch.int8)`。

### 5.2 写入

每次产生新 K/V 或 MLA/DSA 状态时：

```text
BF16 K/V/state
      ↓ C8 store/reshape operator
packed INT8 cache + scale/quant metadata
      ↓
paged cache block
```

写入路径需要同时覆盖：

- 普通 prefill；
- chunked prefill；
- decode append；
- CUDA/NPU graph capture；
- prefix cache；
- cache block copy/迁移。

### 5.3 读取

SFA kernel 接收：

```text
query
LI 产生的 top-k token/block
packed INT8 SFA cache
scale/quant metadata
```

然后只读取 top-k 对应的 cache，并在 kernel 内完成反量化和 attention。

SFA C8 的价值来自两点：

1. cache 容量下降；
2. sparse attention 读取历史 cache 的带宽下降。

如果在 kernel 外先把选中的 cache 全部转回 BF16，仍可能节省容量，但性能
收益会较弱。

## 6. SGLang 需要改哪些地方

不是增加一个 `--kv-cache-dtype int8` 就够了。最小闭环包含五层：

### 6.1 配置和选择

增加两个独立开关：

```text
enable_sparse_li_c8
enable_sparse_sfa_c8
```

启动时明确打印：

```text
LI cache dtype
SFA cache dtype
实际选择的 attention/indexer backend
不支持时是否 fallback
```

不能因为 checkpoint 名字带 `c8` 就静默启用。

### 6.2 Cache spec 和内存计算

SGLang 的 cache 描述需要表达：

```text
dtype
data layout
scale dtype/layout
每 token/block 占用字节数
LI 和 SFA 是否分别启用 C8
```

否则 scheduler 计算的最大 token 数会不准确。

### 6.3 Cache pool 生命周期

data 与 scale 必须一起参与：

```text
allocate
write/append
prefix reuse
block copy
evict/free
reset
```

任何一步只移动 data、不移动 scale，都会造成静默精度错误。

### 6.4 写入算子

至少需要：

```text
store_li_c8(index_key, slot_mapping)
store_sfa_c8(kv_or_state, slot_mapping)
```

算子负责量化、scale 生成、layout 转换和写入目标 cache slot。

### 6.5 LI/SFA kernel

最后让两个计算 kernel 分别消费：

```text
LI  : INT8 index cache + scale
SFA : packed INT8 main cache + quant metadata
```

没有对应 NPU kernel 时，可以先写反量化 fallback 验证数据流，但不能把
fallback 当作最终性能实现。

## 7. EP32、DP8、TP4 下要注意什么

C8 cache 属于 attention/indexer，不按 EP expert 分片，所以它与 W4 expert
的 `EP32` 没有直接关系。

真正影响 cache ownership 的是 attention 的 DP/TP/CP 配置：

- TP4：cache head/channel 和 scale 如何按 TP rank 分片；
- DP8：每个 DP replica 维护自己的请求和 cache；
- 如果启用 DSA-CP：top-k、packed cache 和 scale 是否需要跨 CP rank
  gather/replicate；
- 如果启用 PD：P 节点向 D 节点传输时，data、scale、dtype 和 layout
  必须完全一致。

因此第一版建议只覆盖当前 PD-mix、无额外 DSA-CP 的运行方式。确认 LI C8
正确后，再扩展到 SFA C8、CP 和 PD 分离。

## 8. 如何确认 C8 真的启用了

“模型能跑”不是证据。至少需要同时满足：

1. cache spec 显示 LI 或 SFA dtype 为 INT8；
2. allocator 实际分配 INT8 data 和对应 scale；
3. 写入路径调用 C8 store/reshape operator；
4. LI/SFA backend 调用能够消费 INT8 cache 的 kernel；
5. 打开 C8 后，同样 token 容量下 HBM 占用下降；
6. 关闭 C8 后明确回到 BF16 cache。

可以增加一次性的启动摘要：

```text
Sparse cache configuration:
  LI C8:  enabled, data=int8, scale=fp16
  SFA C8: disabled, data=bf16
  DSA CP: disabled
```

如果日志里只有 checkpoint 的：

```text
indexer_quant_type=INT8_DYNAMIC
```

这不能证明 cache 是 C8。

## 9. 推荐开发顺序

### 第一步：确定当前实际状态

在已经跑通的 W4A8 镜像里打印 LI/SFA cache 的 dtype、shape 和 allocator
结果，确认当前是：

```text
W4A8 + BF16 LI cache + BF16 SFA cache
```

### 第二步：实现 LI C8

按顺序完成：

```text
配置开关
→ indexer cache spec
→ INT8 data + scale 分配
→ prefill/decode 写入
→ LI kernel 读取
→ top-k 与 BF16 对比
→ prefix cache
```

### 第三步：实现 SFA C8

按顺序完成：

```text
packed main-cache spec
→ 容量计算
→ prefill/decode 写入
→ SFA kernel 读取
→ attention/logits 对比
→ prefix cache
```

### 第四步：扩展分布式能力

最后处理：

```text
DSA context parallel
PD cache transfer
HiCache/offload
MTP
```

## 10. 一句话理解

W4A8 解决的是：

```text
expert FFN 怎么算得更省
```

C8 解决的是：

```text
稀疏 attention 的历史状态怎么存得更省、读得更快
```

C8 的完整实现不是一个 dtype 修改，而是：

```text
cache 描述
+ 内存分配
+ INT8 data/scale 生命周期
+ 写入量化算子
+ LI/SFA 消费 INT8 的算子
+ 并行与 cache 传输
```

对当前 SGLang，最小且可验证的切入点是 **先实现 LI C8，保持 SFA cache
为 BF16**。
