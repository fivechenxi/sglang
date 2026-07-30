# GLM-5.2 W4A8C8 在 Ascend 上的实现原理与 SGLang 适配分析

配套实施步骤、测试矩阵和验收门槛见
[GLM-5.2 W4A8C8 Ascend 适配开发计划](GLM52_W4A8C8_DEVELOPMENT_PLAN.md)。

## 1. 背景与结论

目标模型：

- 已跑通基线：`Eco-Tech/GLM-5.2-w8a8`
- 待适配模型：`Eco-Tech/GLM-5.2-w4a8c8`
- 硬件拓扑：4 台 Ascend 910B，共 32 卡
- SGLang 并行配置：EP32、DP8、TP4
- 线上镜像：`quay.io/ascend/sglang:main-cann9.0.0-910b`

总体结论：

1. W4A8C8 不是一种不可拆分的单体量化方案。
2. W4A8 是 routed expert 的权重与激活计算方案；C8 是稀疏注意力/indexer 相关运行时缓存方案。
3. C8 可以关闭，先以 W4A8 权重配合 BF16 cache 运行。
4. SGLang 已具备大部分 W4A8 MoE 基础设施，补齐特定 checkpoint 支持的改动量相对有限。
5. 完整实现 vLLM Ascend 对等的 LI/SFA C8 cache，涉及 cache 生命周期、注意力算子、CP/PD 通信等，改动量明显更大。
6. 当前实际错误：

   ```text
   aclnnGroupedMatmulV5
   bias Dim must be 2, but now is 3
   error 161002
   ```

   已经表明模型进入了 W4A8 GroupedMatmul 执行路径。当前阻塞首先是 `scale_bias` 后处理问题，不是缺少 C8。

---

## 2. W4A8C8 checkpoint 的实际组成

### 2.1 命名含义

通常可将 W4A8C8 分解为：

| 部分 | 含义 |
|---|---|
| W4 | 权重使用 INT4 |
| A8 | 计算输入 activation 动态量化为 INT8 |
| C8 | attention/indexer cache 使用 INT8 |

这里的 C8 不是 W4 GroupedMatmul 的必需输入，也不是 checkpoint 中所有 INT8 权重的统称。

### 2.2 W8A8 与 W4A8C8 量化描述对比

对两个 ModelScope checkpoint 的 `quant_model_description.json` 统计结果：

| checkpoint | 主要量化条目 |
|---|---|
| `GLM-5.2-w8a8` | 约 173,502 个 `W8A8_DYNAMIC` |
| `GLM-5.2-w4a8c8` | 约 233,472 个 `W4A8_DYNAMIC`、1,726 个 `W8A8_DYNAMIC`、17 个 `INT8_DYNAMIC` indexer |

W4A8C8 checkpoint 的关键元数据：

```text
version = 1.0.0
group_size = 0
model_quant_type = W8A8_DYNAMIC
indexer_quant_type = INT8_DYNAMIC
is_rot_used = true
```

其中：

- `version=1.0.0`：采用新版 packed INT4 checkpoint layout。
- `group_size=0`：W4 expert 使用 per-channel 量化，而不是 per-group 量化。
- `is_rot_used=true`：checkpoint 包含 QuaRot rotation 数据。
- 顶层 `model_quant_type=W8A8_DYNAMIC` 不代表所有层都是 W8；具体量化类型必须按每层条目判断。

### 2.3 哪些部分是 W4

GLM-5.2 的 256 个 routed experts 中：

```text
gate_proj
up_proj
down_proj
```

使用 `W4A8_DYNAMIC`。

以下部分仍主要是 W8A8 或 BF16：

- shared expert
- 普通 attention projection
- 部分 dense MLP
- embedding、norm、lm_head
- 其他明确标记为 `FLOAT` 的模块

因此不能将整个模型统一交给一种 W4 Linear kernel。运行时必须支持混合方案：

```text
routed experts → W4A8
shared expert  → W8A8_DYNAMIC
普通 Linear   → W8A8_DYNAMIC 或 FLOAT
```

### 2.4 MTP revision

模型提供两个 revision：

| revision | MTP |
|---|---|
| `master` | MTP 也量化 |
| `nomtpquant` | MTP 保持 BF16 |

首次适配建议：

1. 先关闭 speculative decoding。
2. 或先使用 `nomtpquant` 排除量化 MTP 的影响。
3. target model 基础精度稳定后，再验证 `master` 的量化 MTP。

---

## 3. W4A8 的计算原理

## 3.1 权重存储

新版 ModelSlim `version=1.0.0` 将两个 INT4 数值打包在一个 INT8 存储单元中。

逻辑上：

```text
两个 4 bit 权重
    ↓ pack
一个 int8 storage
```

NPU grouped matmul 接口最终通常使用 `int32` 作为 packed INT4 容器。加载后不是重新进行数值量化，而是：

```text
checkpoint int8 packed storage
    ↓ transpose/layout transform
NPU-oriented layout
    ↓ 每 4 个 int8 重新解释
packed int32
```

核心操作类似：

```python
weight = weight.transpose(...).contiguous()
weight = npu_format_cast(weight)
weight = weight.view(torch.int32).contiguous()
```

`view(torch.int32)` 是存储重解释，不是将 INT8 数值转换为普通 INT32 数值。

### 3.2 新版 expert 权重形状

对新版 packed checkpoint：

```text
w13 weight:
[experts, intermediate_size, hidden_size]

w2 weight:
[experts, hidden_size / 2, intermediate_size]
```

这里的输出维度已经因为两个 INT4 打包进一个 INT8 而减半。

旧版未预打包 checkpoint 的创建形状不同，因此实现必须读取：

```python
new_quant_version = quant_description["version"] == "1.0.0"
```

不能永远假设 checkpoint 都是新版，也不能把新版 packed weight 当成普通 W8 weight。

### 3.3 Activation A8

W4A8 MoE 的输入 activation 在运行时动态量化：

```text
BF16 hidden states
    ↓ per-token dynamic quant
INT8 hidden states + per-token scale
    ↓
W4A8 grouped matmul
```

这与 W8A8 的共同点是 activation 均可动态量化为 INT8；主要差异在权重 layout、scale 处理和对应的 grouped matmul 参数。

### 3.4 Scale 与 scale_bias

新版 W4A8 checkpoint 包含：

```text
weight
weight_scale
weight_offset
scale_bias
```

对于 routed MoE，运行时 fused weight 一般分为：

```text
w13 = gate_proj + up_proj
w2  = down_proj
```

算子需要每个 local expert 对应的二维 bias：

```text
[local_expert_num, output_dim]
```

但 checkpoint/加载参数可能是三维：

```text
[local_expert_num, output_dim, 1]
```

需要在权重后处理阶段转换：

```python
scale_bias = (
    scale_bias
    .transpose(1, 2)
    .contiguous()
    .sum(dim=1)
)
```

转换结果：

```text
[E, N, 1]
  → [E, 1, N]
  → [E, N]
```

### 3.5 EP 下的权重分布

当前拓扑：

```text
总卡数 = 32
EP = 32
DP = 8
TP = 4
```

对于 256 个 routed experts，理想情况下：

```text
256 / 32 = 8 local experts per EP rank
```

需要严格区分：

- global expert ID：checkpoint 中的 `0..255`
- physical/local expert ID：当前 rank 本地 tensor 中的 `0..7`

加载过程必须完成：

```text
global expert ID
    ↓ expert ownership mapping
是否属于当前 EP rank
    ↓
local expert slot
```

若直接用 global expert ID 写入只有 8 个 slot 的参数，会出现：

- 越界或 shape mismatch
- 未初始化 expert 权重
- expert 路由到错误权重
- 精度严重异常

---

## 4. vLLM Ascend 的 W4A8 改造

本地仓库：

```text
/Users/fivechen/Documents/Workspace/vllm-ascend
```

核心文件：

```text
vllm_ascend/quantization/modelslim_config.py
vllm_ascend/quantization/methods/w4a8.py
```

### 4.1 ModelSlim scheme dispatch

vLLM Ascend 根据每层量化描述选择：

```text
W4A8_DYNAMIC → W4A8 Linear/MoE scheme
W8A8_DYNAMIC → W8A8 scheme
FLOAT         → unquantized scheme
```

同时维护 GLM/DeepSeek 类模型的 packed module mapping，例如：

```text
gate_proj + up_proj → gate_up_proj/w13
每个 expert 的 gate/up/down → fused MoE 参数
q_a_proj + kv_a_proj → fused projection
```

### 4.2 显式解析 checkpoint 元数据

vLLM W4A8 实现显式读取：

```python
self.group_size = quant_description.get("group_size", 256)
self.new_quant_version = quant_description.get("version") == "1.0.0"
```

并据此决定：

- per-channel 或 per-group
- 旧版或新版 weight shape
- 是否需要在线重新 pack
- 是否创建 `scale_bias`

### 4.3 EP/TP 语义

vLLM Ascend 使用：

```python
effective_tp = (
    1
    if enable_expert_parallel
    else tensor_parallel_world_size
)
```

原因是启用 EP 后，routed expert 不再按普通 TP Linear 的方式分片。

新版 W4 权重还存在限制：

```python
if new_quant_version and effective_tp > 16:
    raise ValueError(...)
```

当前 SGLang EP32/DP8/TP4 拓扑不等同于 MoE TP32。对 routed expert 来说，effective TP 应按 EP 语义处理，通常为 1。

### 4.4 GroupedMatmul 执行

W4A8 fused MoE 执行链：

```text
router/top-k
    ↓
token dispatch
    ↓
activation dynamic INT8 quant
    ↓
W13 W4A8 grouped matmul
    ↓
SiLU + mul
    ↓
W2 W4A8 grouped matmul
    ↓
token combine
```

vLLM Ascend 还覆盖了：

- MC2/MegaMoE
- fused dispatch/FFN/combine
- EPLB
- shared expert overlap
- 不同 A2/A3/A5 kernel 路径

这些大部分是通用 W4A8 能力，并非 GLM-5.2 专属代码。

---

## 5. SGLang 当前 W4A8 能力与缺口

本地仓库：

```text
/Users/fivechen/Documents/Workspace/sglang
```

已有实现：

```text
python/sglang/srt/layers/quantization/modelslim/modelslim.py
python/sglang/srt/layers/quantization/modelslim/schemes/modelslim_w4a8_int8_moe.py
python/sglang/srt/hardware_backend/npu/quantization/moe_methods.py
python/sglang/srt/layers/moe/moe_runner/ascend.py
```

### 5.1 已具备能力

SGLang 已具备：

- `W4A8_DYNAMIC` MoE scheme dispatch
- 新版 packed W4 expert 参数形状
- activation dynamic INT8 quant
- NPU weight layout 转换
- packed INT4 → int32 storage reinterpret
- Ascend GroupedMatmul
- W13/W2 两阶段 MoE
- DeepEP/MoE Runner
- `scale_bias`
- W4A8 bias shape 修复
- W4A8 性能修复

因此 W4A8 不是从零开发。

### 5.2 元数据解析不完整

当前 SGLang W4 scheme 使用默认参数：

```python
group_size = 0
tp_size = 1
activation_use_clip = False
```

但没有完整地从 checkpoint 读取并传入：

```text
version
group_size
activation_use_clip
EP enabled
effective TP size
```

对当前 checkpoint：

```text
group_size=0
version=1.0.0
```

恰好接近代码的隐含假设，但应当改成显式契约，而不是依赖默认值。

建议：

```python
new_quant_version = quant_description.get("version") == "1.0.0"
group_size = quant_description.get("group_size", 0)
effective_tp = 1 if enable_ep else tp_size
```

同时对暂不支持的组合尽早报错。

### 5.3 mixed routed/shared expert

W4A8C8 checkpoint：

```text
routed experts = W4A8
shared expert  = W8A8
```

如果启用 shared expert fusion/overlap，必须确认：

- routed 和 shared expert 没有被错误合并为同一种量化 scheme
- W4 和 W8 的 scale/bias contract 不混用
- output dtype 与 residual combine 一致

首次启动建议关闭 shared expert overlap/fusion。

### 5.4 缺少精确 checkpoint 回归

目前需要补充至少以下测试：

- GLM-5.2 W4A8C8 参数创建
- `version=1.0.0` packed shape
- EP32 global-to-local expert mapping
- W13/W2 `scale_bias` 二维化
- routed W4 + shared W8 混合执行
- 单请求 eager 精度
- MTP `master`
- BF16 MTP `nomtpquant`

---

## 6. 当前 GroupedMatmul bias 报错分析

实际错误：

```text
aclnnGroupedMatmulV5
bias Dim must be 2, but now is 3
error 161002
```

调用链本质上是：

```text
ModelSlim W4A8 expert
    ↓
NPUW4A8Int8MoEMethod.apply
    ↓
_get_bias_args
    ↓
GroupedMatmul.forward
    ↓
aclnnGroupedMatmulV5
```

GroupedMatmul 接收到的 bias 来自：

```python
w13_scale_bias
w2_scale_bias
```

或 fallback：

```python
w13_weight_bias
w2_weight_bias
```

### 6.1 已有对应修复

SGLang PR：

```text
#31707
commit 1f637a65b933be9582aa28144b0a3c96cc52573a
[NPU] bugfix for W4A8MoE bias 3D dimension mismatch problem
```

其核心是调用 `_update_bias()` 将三维 `scale_bias` 转换为二维。

后续相关修复：

```text
commit 108182cb8199190a66f14c04ff8f0cc669ea7af2
Fix w4a8 MoE performance degradation
```

### 6.2 浮动镜像风险

线上镜像使用：

```text
quay.io/ascend/sglang:main-cann9.0.0-910b
```

`main` 是浮动 tag，不能根据名称确认源码 commit。四台机器还可能因为 `IfNotPresent` 或本地缓存使用不同 digest。

必须在每个节点检查：

```bash
docker image inspect quay.io/ascend/sglang:main-cann9.0.0-910b \
  --format '{{index .RepoDigests 0}} {{.Created}}'
```

或：

```bash
crictl images --digests | grep 'quay.io/ascend/sglang'
```

生产环境建议固定：

```text
quay.io/ascend/sglang@sha256:<digest>
```

### 6.3 运行容器代码检查

检查实际导入路径：

```bash
python -c 'import sglang; print(sglang.__file__)'
```

检查是否包含修复：

```bash
python -c 'import inspect; from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW4A8Int8MoEMethod; print(inspect.getsource(NPUW4A8Int8MoEMethod._update_bias))'
```

检查调用是否存在：

```bash
python -c 'import inspect; from sglang.srt.hardware_backend.npu.quantization.moe_methods import NPUW4A8Int8MoEMethod; print(inspect.getsource(NPUW4A8Int8MoEMethod.process_weights_after_loading))'
```

### 6.4 推荐防御

正确做法是在 `process_weights_after_loading()` 永久将 bias 转成二维。

另外可在 `_get_bias_args()` 增加最终防御：

```python
if bias is not None and bias.dim() == 3:
    if bias.shape[1] == 1:
        bias = bias.squeeze(1)
    elif bias.shape[2] == 1:
        bias = bias.squeeze(2)

if bias is not None and bias.dim() != 2:
    raise ValueError(
        f"{weight_prefix} GMM bias must be 2D, got {tuple(bias.shape)}"
    )
```

需要记录：

```text
weight_prefix
weight.shape
scale.shape
bias.shape
local expert count
rank/EP/DP/TP
```

EP32 下，bias 第一维通常应约为 8：

```text
w13 bias: [8, w13_output]
w2 bias:  [8, hidden_size]
```

如果第一维是 256，则除了三维 bias 外，还存在 global/local expert 映射问题。

---

## 7. C8 的原理

### 7.1 C8 优化的对象

C8 主要优化运行时 attention/indexer cache，而不是 routed expert 权重。

GLM-5.2 的稀疏注意力涉及：

- LI：Lightning Indexer
- SFA：Sparse Flash Attention
- DSA：DeepSeek Sparse Attention
- Indexer Share

可以将 cache 从 BF16 压缩为 INT8：

```text
BF16 cache element: 2 bytes
INT8 cache element: 1 byte
```

理论上 cache 数据部分可节省约一半显存和通信带宽，但需要额外 scale 和量化/反量化操作。

### 7.2 Cache 生命周期

完整 C8 不是简单改一个 dtype，需要覆盖：

```text
量化 scale 生成/加载
    ↓
C8 cache 容量计算
    ↓
cache pool/block 分配
    ↓
prefill 写入 INT8 cache
    ↓
decode 按 slot/block 读取
    ↓
SFA/LI 算子消费 INT8 + scale
    ↓
prefix cache 复用
    ↓
CP/PD cache 传输
```

每一环都需要对 dtype、shape、layout 和 scale ownership 达成一致。

### 7.3 LI C8 与 SFA C8

vLLM Ascend 将 C8 拆成：

```text
enable_sparse_li_c8
enable_sparse_sfa_c8
```

典型 GLM-5.2 配置：

```json
{
  "enable_sparse_li_c8": true,
  "enable_sparse_sfa_c8": false,
  "c8_enable_reshape_optim": false
}
```

这说明 LI 与 SFA 可以独立启停，不应将其实现为一个不可拆分的全局开关。

### 7.4 C8 scale

INT8 cache 必须配套 scale：

```text
float value ≈ int8 value × scale
```

实现需要明确：

- per-tensor、per-channel 或其他 scale 粒度
- scale 是 checkpoint 固定值还是 runtime dynamic
- prefill/decode 是否共享 scale
- CP/PD 传输时 scale 是否一并传输
- prefix cache block 对 scale 的所有权

scale 管理错误可能不会立即 crash，但会造成严重精度退化。

---

## 8. vLLM Ascend 的 C8 改造范围

重要相关提交包括：

| 提交/PR | 作用 |
|---|---|
| `#7029` | 早期 DSV3.2/GLM5 W8A8C8 支持，包含自定义 indexer quant kernel、SFA、cache interface 和 model runner |
| `#10060` | DSA-CP sparse C8 适配 |
| `#11228` | SFA C8 packed KV cache layout |
| `#11870` | DCP replicated indexer 下的 C8 SFA |
| `#12470` | 将统一 C8 拆分为 LI C8 和 SFA C8 |

修改范围覆盖：

```text
Ascend configuration
SFA backend
SFA context parallel
KV cache interface
cache allocator/device op
model runner
CP all-gather
PD/Mooncake transfer
自定义 NPU kernel
单元测试与多卡测试
```

因此完整 C8 属于独立功能项目，不应与“先让 W4 checkpoint 启动”绑定交付。

---

## 9. C8 是否可以关闭

可以关闭。

在 vLLM Ascend 中可设置：

```json
{
  "enable_sparse_li_c8": false,
  "enable_sparse_sfa_c8": false
}
```

关闭后可形成：

```text
routed expert weights = W4
MoE activation = INT8
sparse attention cache = BF16
```

需要区分：

1. checkpoint 中的 `W4A8_DYNAMIC` expert 权重必须支持。
2. checkpoint 中的 INT8 indexer 权重仍需正确加载。
3. runtime LI/SFA cache 是否使用 INT8 可以关闭。

关闭 C8 的主要代价：

- cache 显存占用更大
- 长上下文可用容量下降
- cache 读写和通信量增加
- 性能可能低于 vLLM Ascend C8 配置

但关闭 C8 不应阻止模型启动。

---

## 10. SGLang 完整支持 C8 所需改造

若追平 vLLM Ascend，需要至少覆盖：

### 10.1 配置层

- `enable_sparse_li_c8`
- `enable_sparse_sfa_c8`
- `c8_enable_reshape_optim`
- checkpoint C8 metadata 解析
- 明确默认值及硬件限制

### 10.2 Cache interface

- C8 cache dtype/layout
- packed byte cache
- block size 与 alignment
- cache 容量计算
- scale buffer
- cache group 一致性校验

### 10.3 写入与读取

- prefill store C8
- decode append C8
- LI 读取 C8
- SFA 读取 C8
- scale 应用
- graph/eager 一致性

### 10.4 并行与通信

- DSA-CP
- replicated indexer
- all-gather packed cache
- EP/DP/TP 组合
- prefix cache
- HiCache
- PD/Mooncake transfer

### 10.5 Kernel

需要确认现有 CANN/torch-npu 是否已经提供所有必需算子，否则还需：

- `sgl-kernel-npu` 自定义 kernel
- meta/shape inference
- tiling
- graph capture 支持
- A2 910B 专项验证

### 10.6 测试

- eager/full graph
- prefill/decode
- prefix cache
- CP
- PD
- MTP
- 长上下文
- scale 精度
- 多机一致性

---

## 11. 推荐实施路线

## 阶段一：W4A8 基础启动

目标：

```text
W4 routed experts + A8 activation + BF16 cache
```

建议关闭：

- C8
- MTP
- graph
- EPLB
- prefix cache
- HiCache
- PD
- DSA CP/cache split
- shared expert overlap/fusion

工作内容：

1. 修复/验证 `scale_bias` 二维化。
2. 显式解析 `version=1.0.0`、`group_size=0`。
3. 验证 EP32 local expert 数为 8。
4. 验证 global-to-local expert loader。
5. 验证 routed W4 + shared W8。
6. 单请求 eager decode。

预计改动：

```text
约 100–300 行代码
约 100–200 行测试
```

若仅当前 bias 问题，可能只需几十行或更新到已包含修复的镜像。

## 阶段二：恢复生产并行与 MTP

按顺序恢复：

1. DeepEP 基础模式
2. shared expert overlap
3. graph
4. prefix cache
5. MTP `nomtpquant`
6. MTP `master`
7. 长上下文
8. PD

预计额外改动：

```text
约 200–500 行
```

实际耗时主要来自四机调试和精度回归。

## 阶段三：LI C8

优先实现 LI C8，因为 vLLM GLM-5.2 常用配置是：

```text
LI C8 = on
SFA C8 = off
```

先覆盖：

- indexer cache
- scale
- prefill/decode
- EP32/DP8/TP4
- prefix cache

## 阶段四：SFA C8 与完整分布式能力

最后扩展：

- SFA C8
- DSA-CP
- packed cache all-gather
- PD/Mooncake
- HiCache
- 1M context

完整 C8 预计是上千行级别的独立项目。

---

## 12. 精度验证方案

### 12.1 验证矩阵

建议至少包含：

| 实现 | 权重 | Cache | 用途 |
|---|---|---|---|
| SGLang W8A8 | W8A8 | 当前基线 | 已知正确基线 |
| vLLM Ascend W4A8C8 | W4A8 | C8 off | W4 实现参考 |
| SGLang W4A8C8 checkpoint | W4A8 | BF16 | 待验证目标 |
| vLLM Ascend W4A8C8 | W4A8 | C8 on | C8 最终参考 |

判断 SGLang W4 实现是否正确时，最重要的参考是：

```text
同一 checkpoint、同一 tokenizer、C8 关闭的 vLLM Ascend
```

不能只要求 W4 与 W8 token-by-token 完全一致，因为量化会改变临界 logits。

### 12.2 最小输出验证

固定：

```text
temperature = 0
top_p = 1
seed 固定
MTP off
C8 off
eager
```

使用 50–100 条固定 prompt，检查：

- HTTP 成功率
- token IDs
- 输出长度
- 乱码
- 重复
- 空输出
- NaN/Inf
- 各 DP rank 一致性

### 12.3 Logits 对比

对固定输入保存：

- prefill 最后一个 token logits
- 第一个 decode token logits

比较：

```text
cosine similarity
max absolute error
mean absolute error
top-1 agreement
top-5 overlap
```

若差异异常，从第一层开始对比：

```text
MoE 输入
W13 输出
SiLU/mul 输出
W2 输出
MoE combine 输出
shared expert 输出
layer residual
```

### 12.4 公共数据集

建议顺序：

1. GSM8K 小样本冒烟
2. AIME 2025/2026
3. GPQA Diamond
4. MMLU-Pro
5. HumanEval/EvalPlus
6. Tool calling
7. 长上下文

必须保持一致：

- dataset revision
- prompt/template
- tokenizer
- reasoning 模式
- `max_new_tokens`
- temperature
- stop tokens
- scoring 脚本

建议验收：

```text
SGLang W4 相对 vLLM W4：
主要指标绝对差不超过约 0.5–1.0 个百分点
```

AIME 样本少、波动较大，应结合重复测试或置信区间。

### 12.5 MTP 验证

基础 target model 通过后，再分别验证：

- `nomtpquant`
- `master`

记录：

- 最终任务精度
- acceptance length
- MTP 开关前后输出异常率
- prefix cache 影响
- 各 DP rank 一致性

---

## 13. 工作量评估

| 目标 | 工作量判断 |
|---|---|
| 修复当前三维 bias | 很小，已有对应上游修复 |
| W4A8 checkpoint 基础启动 | 小到中，约 100–300 行 |
| EP32/DP8/TP4 + DeepEP + MTP 稳定运行 | 中等，约 300–700 行及多机验证 |
| LI C8 | 中到大，需要独立设计 cache/scale/算子路径 |
| 完整 LI/SFA C8 + CP/PD/HiCache | 大，上千行级别并可能涉及 NPU kernel |

总体建议：

```text
先交付 W4A8 + BF16 cache
    ↓
验证显存收益、精度和稳定性
    ↓
恢复 MTP 与生产优化
    ↓
将 LI C8 单独立项
    ↓
最后实现 SFA C8/CP/PD
```

这样可以避免让 C8 的大规模 cache 改造阻塞 W4 权重带来的显存收益。
