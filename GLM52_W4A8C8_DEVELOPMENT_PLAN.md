# GLM-5.2 W4A8C8 Ascend 适配开发计划

## 1. 目标与范围

本计划面向以下部署环境：

- 模型：`Eco-Tech/GLM-5.2-w4a8c8`
- 已知可运行基线：`Eco-Tech/GLM-5.2-w8a8`
- 硬件：4 台 Ascend 910B，共 32 卡
- 并行配置：EP32、DP8、TP4
- 当前镜像：`quay.io/ascend/sglang:main-cann9.0.0-910b`
- 开发基线：SGLang `main`

近期目标是先让 checkpoint 以 W4A8 expert、非 C8 cache 的方式正确运行并通过精度验证。C8 作为独立阶段实现，不作为 W4A8 首次跑通的前置条件。

本次文档 PR 不包含运行时代码修改。后续代码按阶段拆分成小 PR，避免把 checkpoint 加载、MoE 算子、cache 生命周期和分布式问题混在一次评审中。

## 2. 当前状态

### 2.1 已知错误

当前 W4A8C8 checkpoint 在启动或首次执行 MoE 时出现：

```text
aclnnGroupedMatmulV5
bias Dim must be 2, but now is 3
error 161002
```

该错误说明模型已经进入 NPU W4A8 GroupedMatmul 路径，传给算子的 `scale_bias` 仍为三维，而算子要求二维。

### 2.2 `main` 已有相关修复

在开始写新实现前，必须先验证以下已经进入上游历史的修复：

| 变更 | Commit | 作用 |
|---|---|---|
| #31707 | `1f637a65b933be9582aa28144b0a3c96cc52573a` | 在 W4A8 MoE 权重后处理阶段将 3D `scale_bias` 归一化为 2D |
| #32113 | `108182cb8199190a66f14c04ff8f0cc669ea7af2` | 修正 bias、权重 layout 和 INT4 packing 的连续性处理，解决性能退化 |

因此，第一步不是再次实现同样的 `transpose/sum`，而是回答以下问题：

1. 当前线上镜像是否包含这两个 commit。
2. GLM-5.2 是否确实选择了 `NPUW4A8Int8MoEMethod`。
3. `w13_scale_bias` 和 `w2_scale_bias` 是否都执行了后处理。
4. EP 权重加载后，local expert 维度和算子收到的二维 bias 是否一致。
5. #32113 的后续修改是否改变了 #31707 对 GLM-5.2 checkpoint 的实际效果。

## 3. 开发原则

1. **W4A8 与 C8 解耦。** 先用普通 BF16 cache 验证 W4A8；C8 单独设计、测试和评审。
2. **先固定可复现基线。** 记录镜像 digest、SGLang SHA、CANN、torch/torch_npu、ATB/自定义算子版本。
3. **先单 rank 数据正确，再扩大并行规模。** 优先验证 tensor shape、dtype、layout、expert 映射，再运行 32 卡。
4. **保留 W8A8 回归基线。** 相同 prompt、sampling 参数和并行配置下，对比 W8A8 与 W4A8。
5. **不在热路径保留调试同步。** shape/assert/logging 使用显式 debug 开关，验证完成后移除或默认关闭。
6. **小 PR 交付。** 每个 PR 只解决一个能够独立验证的问题。

## 4. Phase 0：固定基线并复现

### 4.1 记录环境

在 4 台机器上收集：

```bash
docker inspect --format='{{index .RepoDigests 0}}' \
  quay.io/ascend/sglang:main-cann9.0.0-910b

python -c "import importlib.metadata as m; print(m.version('sglang'))"
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__)"
pip freeze
```

同时记录容器 OCI labels 中的 revision、version 和 build time。如果镜像没有源码 SHA label，需要从 wheel metadata、容器源码目录或构建流水线反查准确 commit。

### 4.2 最小复现材料

保存：

- 完整启动命令和全部量化相关参数
- 4 个节点的 rank table/网络配置摘要
- 从权重加载开始到报错位置的完整日志
- 模型 `config.json`
- `quant_model_description.json`
- checkpoint revision（`master` 或 `nomtpquant`）

### 4.3 基线判断

- 若镜像不包含 #31707/#32113：先构建最新 `main` 镜像复测。
- 若最新 `main` 已经跑通：后续工作转为回归测试、精度和 C8，不再提交重复修复。
- 若最新 `main` 仍报 3D bias：进入 Phase 1 定位 GLM-5.2 特有加载路径。

Phase 0 完成标准：

- 任何开发者能用固定 digest 和固定 SHA 重现同一结果。
- 明确错误发生在权重加载、warmup、prefill 或 decode 的哪一阶段。
- 明确模型选择的 quant method、MoE runner 和 grouped matmul 接口版本。

## 5. Phase 1：W4A8 权重加载与后处理

### 5.1 增加定点诊断

在 `NPUW4A8Int8MoEMethod` 的权重创建、加载后处理和 `apply` 边界临时采集：

```text
tensor name
global/local expert count
shape before loading
shape after loading
shape after transpose
shape after npu_format_cast
shape after int32 packing
scale/offset/scale_bias shape and dtype
contiguous status
selected quant method
```

只在一个 rank 输出完整信息，其他 rank 仅在断言失败时输出，避免 32 卡日志失控。

### 5.2 验证关键不变量

对 `w13` 与 `w2` 分别验证：

- packed checkpoint layout 与参数创建 shape 一致。
- `version=1.0.0` 走新版预打包路径。
- `group_size=0` 被解释为 per-channel，而非非法 group。
- `scale_bias` 进入 `aclnnGroupedMatmulV5` 前严格为 2D。
- `weight` 在 transpose 后先 contiguous，再执行 NPU format cast。
- INT8 packed storage 以 INT32 重解释后 shape 与算子约定一致。
- shared expert 的 W8A8 路径不被 routed expert 的 W4 逻辑污染。

建议在 Python 边界增加带上下文的显式校验：

```python
assert scale_bias is None or scale_bias.ndim == 2
assert packed_weight.dtype == torch.int32
assert packed_weight.is_contiguous()
```

正式代码中的异常信息应包含 `w13/w2`、shape、expert 数和 quant version，便于线上定位。

### 5.3 EP32 映射

256 个 routed experts、EP32 时每 rank 预计持有 8 个 local experts。需要验证：

```text
global expert id → local physical expert id
checkpoint shard → local weight slot
scale/offset/bias → 相同 local expert slot
```

不能只验证 weight；weight、scale、offset 和 bias 必须使用同一映射。

Phase 1 完成标准：

- W4A8 权重加载与 warmup 完成。
- GroupedMatmul 不再收到 3D bias。
- 所有 32 个 EP rank 的 local expert 数与元数据一致。
- W8A8 checkpoint 回归不受影响。

## 6. Phase 2：W4A8 计算正确性

### 6.1 算子级参考对比

从真实 checkpoint 抽取少量 expert 的：

- packed INT4 weight
- weight scale/offset
- scale bias
- BF16 activation

建立 BF16/FP32 反量化参考实现，对比：

1. activation 动态 INT8 量化结果；
2. 第一段 `w13` grouped matmul；
3. activation/SwiGLU；
4. 第二段 `w2` grouped matmul；
5. routed expert 聚合结果。

验证维度应覆盖：

- 单 token 与多 token
- 单 expert 与多个 local experts
- token 数为 0 的 local expert
- `w13` 与 `w2`
- prefill 与 decode 典型 shape

### 6.2 模型级逐层定位

固定 prompt、关闭随机采样与 speculative decoding，对比：

```text
W8A8 SGLang
W4A8 SGLang
W4A8 vLLM Ascend（可用时）
```

逐层采样以下统计：

- hidden state max/mean/std
- NaN/Inf
- cosine similarity
- normalized RMSE
- top-k logit overlap

如果最终输出异常，使用二分法确定首个明显偏离的 layer，再下钻到 attention、shared expert 或 routed expert。

Phase 2 完成标准：

- 无 NaN/Inf。
- 算子级误差符合 INT4/INT8 量化预期。
- 固定输入可以稳定复现固定输出。
- 与可用参考后端相比，首个 token 和后续 decode 不存在结构性偏差。

## 7. Phase 3：分布式稳定性与性能

按以下阶梯扩容，能用更小模型或裁剪层验证时优先使用：

| 阶段 | 目的 |
|---|---|
| 单卡/单 rank | 权重 shape、packing、单算子正确性 |
| 单机少量 rank | TP/DP/EP 基础通信和 expert 映射 |
| 4 机 32 卡，低并发 | 完整 EP32/DP8/TP4 启动与正确性 |
| 4 机 32 卡，目标并发 | 性能、稳定性和长时间运行 |

重点观察：

- 每 rank 显存/HBM 分布
- 权重加载时间和 warmup 时间
- prefill/decode latency
- TTFT、TPOT、throughput
- HCCL timeout、rank 间 shape 不一致
- 空 expert、极端路由和负载不均
- 多轮请求后的内存增长

与同环境 W8A8 对比，W4A8 至少应证明权重/HBM 收益；若吞吐明显下降，优先检查是否发生 layout copy、fallback 或每步重复 packing。

Phase 3 完成标准：

- EP32/DP8/TP4 可重复启动。
- 短请求、长上下文和连续批处理稳定。
- 没有 rank 特有错误、死锁或持续内存增长。
- 给出 W8A8/W4A8 的容量和性能对比报告。

## 8. Phase 4：端到端精度验收

### 8.1 确定性回归集

建立 50–200 条固定样本，覆盖：

- 中文与英文问答
- 数学、代码和推理
- 长上下文检索
- function/tool call（如部署需要）
- 多轮对话
- 容易触发不同 expert 路由的主题

固定：

```text
temperature=0
top_p=1
seed
max_new_tokens
chat template
tokenizer revision
```

记录 token IDs、首 token logits/top-k、最终文本和终止原因。

### 8.2 数据集评测

根据实际业务选择公开集和内部集。至少保留：

- 业务代表性内部集
- 数学/推理集
- 长上下文集
- 生成质量人工抽检集

比较：

```text
GLM-5.2 W8A8 SGLang → 当前生产基线
GLM-5.2 W4A8 SGLang → 待验收实现
GLM-5.2 W4A8 vLLM Ascend → 交叉参考（条件允许）
```

量化模型不应以逐 token 完全一致作为唯一标准。验收阈值应在跑出基线后冻结，指标包括任务分数相对下降、输出异常率、拒答率和格式正确率。

### 8.3 MTP

第一次验收关闭 speculative decoding，或使用 `nomtpquant` revision。target model 通过后再单独启用量化 MTP，测量：

- acceptance rate
- 任务精度变化
- token/s 收益
- 首个发生差异的位置

Phase 4 完成标准：

- 量化精度阈值经团队确认并固化。
- 关键业务集不存在不可接受回退。
- 结果包含模型 revision、运行 SHA、镜像 digest 和全部推理参数。

## 9. Phase 5：C8 cache 独立适配

### 9.1 第一版默认关闭 C8

W4A8 首次合入时使用现有普通 cache 路径，不读取或启用 C8 特有配置。必须保证：

- W4A8 checkpoint 不因 `indexer_quant_type=INT8_DYNAMIC` 被错误拒绝。
- 关闭 C8 时不创建 INT8 cache/scale。
- attention/indexer 使用现有已验证路径。
- 启停方式明确，日志中输出实际 cache dtype/backend。

### 9.2 C8 实现拆分

C8 适配至少分成以下子任务：

1. **配置与能力判断**
   - 识别 checkpoint 的 C8/indexer 元数据。
   - 校验硬件、CANN 和 attention backend 是否支持。
   - 不支持时显式 fallback 或报错，不能静默使用错误 layout。
2. **cache 分配与生命周期**
   - 定义 INT8 cache layout。
   - 定义 per-tensor/per-channel/per-token scale 的存储。
   - 覆盖 allocate、write、append、gather、evict、prefix reuse 和 reset。
3. **写入量化**
   - 将 BF16/FP16 index/cache state 量化为 INT8。
   - 正确处理 scale、饱和、异常值和 padding。
4. **attention/indexer 读取**
   - 接入支持 C8 的 LI/SFA/DSA kernel。
   - 处理 prefill/decode 不同路径。
   - 明确 fallback 是否需要反量化。
5. **并行与传输**
   - TP/DP/CP 下 cache ownership。
   - 若后续启用 PD，覆盖 cache transfer 的 dtype、shape、scale 和协议版本。
6. **观测与回归**
   - 启动日志展示 C8 是否真实启用。
   - 增加 cache dtype/容量指标和精度测试。

### 9.3 C8 验收

对比 C8 关闭和开启：

- 相同输入的 logits/任务精度
- HBM 占用
- 可支持最大上下文/并发
- TTFT、TPOT、吞吐
- prefix cache 命中场景
- 长上下文误差是否随长度累积

C8 只有在精度和稳定性通过后才能成为推荐配置；首版不建议默认启用。

## 10. 建议 PR 拆分

### PR 0：分析与计划

内容：

- W4A8/C8 原理和差异分析
- 已知错误与上游修复状态
- 分阶段开发与精度验证计划

不修改运行时代码。

### PR 1：GLM-5.2 W4A8 加载兼容与单元测试

仅在最新 `main` 复现失败时创建，内容可能包括：

- GLM-5.2 checkpoint quant metadata 兼容
- `scale_bias` shape 不变量和错误信息
- packed INT4 layout/contiguous 处理
- 不依赖真实大模型权重的单元测试

### PR 2：Ascend GLM-5.2 W4A8 E2E 测试

内容：

- 模型路径常量
- 最小启动/准确性测试
- 条件允许时添加 32 卡性能或 nightly 用例
- W8A8 回归覆盖

### PR 3：C8 配置、cache storage 和 fallback

内容：

- C8 feature flag/capability detection
- INT8 cache 与 scale storage
- C8 关闭路径和显式 fallback
- cache 生命周期单元测试

### PR 4：C8 attention/indexer kernel 集成

内容：

- prefill/decode kernel 接入
- CP/PD（若在目标范围）数据流
- 端到端精度与性能用例

## 11. 风险与决策点

| 风险 | 处理方式 |
|---|---|
| `main` 与线上滚动镜像不一致 | 固定 SHA 和镜像 digest，分别保存复现结果 |
| 上游 bias 修复已解决问题 | 不提交重复代码，转为测试和文档 |
| 只在 EP32 暴露 expert 映射错误 | 增加 global/local expert 映射断言和分 rank 摘要 |
| W4A8 能运行但精度异常 | 算子参考、逐层统计、固定 logits 三层定位 |
| C8 工作量污染 W4A8 交付 | 默认关闭 C8，单独 feature/PR |
| MTP 放大量化问题 | 初次关闭 MTP，target model 通过后单独验收 |
| W4A8 发生隐式 fallback | 日志输出 quant method/kernel，并用性能/HBM 交叉验证 |

需要团队在开发前确认：

1. 首次交付是否明确接受“W4A8 + 非 C8 cache”。
2. C8 是否需要覆盖 PD 分离，还是先支持 PD mix。
3. 精度验收使用哪些内部数据集及允许的相对回退。
4. 最终 PR 是否以最新 `main` 为目标，线上版本通过后续 cherry-pick 维护。

## 12. Definition of Done

W4A8 第一阶段完成需同时满足：

- 最新 `main`、固定依赖和固定镜像可构建。
- GLM-5.2 W4A8 在 4×910B、EP32/DP8/TP4 启动并持续推理。
- 不出现 GroupedMatmul bias 维度、weight layout 或 expert 映射错误。
- W8A8 回归通过。
- 固定 logits 和任务级精度达到冻结阈值。
- 性能/HBM 报告确认没有明显 fallback。
- 启动文档包含模型 revision、参数、限制和 C8 状态。

C8 完成需额外满足：

- cache 全生命周期覆盖 INT8 data 和 scale。
- prefill/decode、prefix reuse、长上下文和目标并行模式通过。
- C8 开关行为明确，关闭时保持 W4A8 可用。
- 精度、容量和性能收益有可复现报告。
