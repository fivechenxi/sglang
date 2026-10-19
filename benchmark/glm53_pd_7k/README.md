# GLM-5.3 / 6000D：已验证增量补丁与复现记录

本目录整理 2026-10-17 的 **NVFP4、4P1D、40 GPU** 实验，不改变 SGLang 默认执行路径。两轮独立有限时长测试的保守总吞吐为 **7014.2 / 7009.3 tokens/s/GPU**。

2026-10-19 补充：[回收、路由均衡与 P→D 错误联动修复](recovery/README.md)。包含独立补丁、真实回收/取消验证和固定源码哈希；不改写下面历史性能结果，GPQA 循环尚未完成归因。

这是固定版本的研究补丁包，**不是新的一体化生产镜像，也不是“本 PR checkout 后直接启动即可得到 7K”**。实测 P 为 v0.5.19 / Torch 2.13，D 为带历史覆盖补丁的 v0.5.17 / Torch 2.11。`patches/` 的应用对象是相应运行源码，不是本分支任意版本的同名文件；移植到当前源码必须重新进行 GPU 正确性及端到端验收。

## 本次整理的内容

| 项目 | 实际改动 | 已有证据与限制 |
|---|---|---|
| P InputSP | attention→MoE 边界采用 attention-TP ReduceScatter → 本地 LayerNorm → TP AllGather；恢复完整 residual | 实际层 hidden/residual 对照；配对 P 测试约 +5.3%，不是整机单项收益 |
| P 完整 KV + PCP4 | 关闭 LayerSplit，保留 PCP4；每个 CP rank 持有完整历史 KV，避免 LayerSplit 历史 KV 广播 | P 代理计时约 +10.9%，包含交接；约多占 4.7 GiB/rank；不是免费容量收益 |
| PD 有界打包 | 大请求分块复用已注册 pack buffer；必须同步传输完成后才能复用，失败立即返回 | 碎片物理页、层间不同字节数、失败不复用的回归；不单独声称 TPS 提升 |
| Gather 预加载 | KV 注册阶段预加载观测到的 gather shape，避免显存紧张时临时加载模块 | 固定 16K prefill profile；不能不检查 buffer 容量就推广到其他参数 |
| D seed 重算 | 关闭跨阶段 draft-extend seed 复用，保留迭代内 IndexShare、target verify 和真实拒绝采样 | draft decode profile 从 1/4 到 4/4 步走 graph；提议/接受率可能变化，不保证逐 token 相同 |
| 精确 DSA Top-K | P 使用社区精确选择修复；D 增加相同 score 时较小 index 优先的确定性 tie-break | D 实际张量 364 文件/1420 行，eager/graph 错误均 0，配对 rank 选择集合一致；属于正确性修复，不单列收益 |
| Router 流式负载计数 | dispatch 起计数，覆盖返回响应头前的阶段；流生命周期接管计数 | mock 验证分配、warm affinity、取消、排空；真实四前缀 P 调度核验 |
| D 容量/graph 配置 | 保留 DCP2；running=88、每 DP graph=22、memory=0.960、extra slots=4 | 物理 150208 / 逻辑 300416 tokens/DP；属于配置调优，不是新 kernel |

InputSP、Top-K 的 `runtime/` 保留已实测的研究 hook 原始字节及 SHA，因此仍有一次性审计日志和 `/research` 等通用挂载点；**未将这些定位代码混入核心热路径**。`operations/` 是从运行文件抽取的原函数，供离线审阅/单测；不应与补丁重复安装。未保留无效的 active-page/TP8-broadcast hook、失败的 MoE tactic、FI 0.7 MLA 等候选。

## 依赖与版本边界

- 叠加基线：[PR #25](https://github.com/fivechenxi/sglang/pull/25)。本 PR 以该分支为审阅基底，不改写其历史。
- 既有 P RS：[PR #26](https://github.com/fivechenxi/sglang/pull/26)，实验对应 `65ae1d2827ea2f027043934a74351aeeec490810`；本次不重复计入收益。
- 既有 DCP/索引池：[PR #27](https://github.com/fivechenxi/sglang/pull/27)。实测 D 是 legacy 源码覆盖，**不能把 #27 当前 v0.5.19 文件整份覆盖到 v0.5.17**。
- #25 后续的 L3/restore 修复不在这次历史镜像的 7K 验收范围内。不要用本目录旧版 `common/conn.py` 补丁覆盖当前 L3 修复。
- P：Torch `2.13.0+cu130` / FlashInfer `0.6.18`；D：Torch `2.11.0+cu130` / FlashInfer `0.6.15` / NCCL `2.28.9`；SM120。
- Router：基于 `0bcd822377da7b5718e674eaf9c870d349424dd1`，最小回移上游 `17c3b259f5be84638617b4dce345c4b10f47bb2b` 的 dispatch guard 修复；不是整批升级 Router。
- P Top-K 来源：社区 [PR #37893](https://github.com/sgl-project/sglang/pull/37893)，固定源码 `f36a7588361ec92351e40e3fb59d56f560ad792b`。CUDA 文件保留原作者/来源说明；D 在其上增加确定性 tie-break。

`manifest.json` 固定运行源码的 pre/post SHA。部分 preimage 是从实测文件逆向去除该单项改动后得到的，清单已明确标注，**不是声称它等于官方 release 原文件**。基础镜像、历史 RS/DCP 覆盖、权重和编译依赖仍需另行提供；本包不含私有镜像、二进制、地址、账号、密钥、用户请求或部署清单。

## 配置摘要

具体数值见 [config.json](config.json)，未包含任何真实 worker URL。

- P×4：每台 8 GPU；TP8 / EP8 / PCP4 / attention-TP2 / DP1；完整 KV，LayerSplit 关闭；FP8 KV；memory=0.88；chunk/max-prefill=16384；每步最多一个新 prefill；running=64；关闭 CUDA graph。
- D×1：8 GPU；TP8 / EP8 / attention-DP4 / attention-TP2 / DCP2 a2a；memory=0.960；running=88；每 DP decode graph=22；reserved decode=512；extra slots=4；inflight transfer=6。
- P/D：MoE a2a=`none`，runner=`flashinfer_cutlass`；真实 EAGLE steps=5、topk=1、draft tokens=6；不使用 synthetic acceptance。
- Router：P/D 均 `cache_aware`，开启 DP-aware，关闭 retries。四组独立可复用前缀需要实际验证分配到四个 P，不能仅看 API 的 cached_tokens。

## 源码检查与离线回归

在仓库根目录执行：

```sh
cd benchmark/glm53_pd_7k
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 check_sources.py --role P --source-root /path/to/isolated-prefill-source
python3 check_sources.py --role D --source-root /path/to/isolated-decode-source
```

`check_sources.py` **只读**：先核对 SHA，再执行 `git apply --check`。任何未知源码立即拒绝；已经是实测 postimage 的文件标记为已应用。不会自动修改安装目录或服务。

此次整理已重新执行 25 项 CPU 单测，4 个运行文件补丁逆向/正向往返后与实测 SHA 一致，Router 补丁在固定公开 base 上检查通过，3 个抽取函数与实测 AST 一致。见 [package_validation.json](results/package_validation.json)。Router 的完整 HTTP mock 脚本在 `integration/router_load_regression.py`；历史 stock/candidate 结果在 `results/router_regression.json`，本次未重新运行 Rust 二进制。脚本仅用于隔离 CPU 容器，`/research` 为测试挂载目录，不能指向线上安装目录。

确认所有 preimage 匹配后，在隔离副本中分别应用相应 patch。对于已应用文件不要重复打；先检查 `git diff`，逐项记录。P 使用 `bounded-pack.patch`、`gather-preload.patch`；D 使用 `replicated-pcp-dcp.patch`、`draft-seed-recompute.patch`。Router 单独在上述固定 commit 上 `git apply --check` 后应用 `router-stream-load.patch`。

只在隔离 worker 启动前加载 hook；不要热替换正在服务的 Python 模块：

- P：将 InputSP hook 和 `input_sp_control.json` 放到隔离挂载目录（控制文件名 `control.json`）。首次用 `mode=baseline, verify_input_sp=true` 对照实际层两个输出，之后再用 candidate。启用 Top-K 需 `GLM53_RESEARCH_EXACT_TOPK=1` 并在加载模型前调用该 hook 的 `install()`；有界传输需 `SGLANG_RESEARCH_DCP_BOUNDED_PACK=1`。
- D：完整 KV 的窄拓扑许可需 `SGLANG_RESEARCH_REPLICATED_PCP_DCP=1`；seed 重算需 `GLM53_RESEARCH_D_RECOMPUTE_SEED=1`；精确 Top-K 需 `GLM53_RESEARCH_D_EXACT_TOPK=1` 并在加载模型前调用其 `install()`。
- hook 默认关闭或显式加载；不支持其他模型/拓扑的泛化承诺。研究原函数名/标记保留，避免把代码清理与新的性能验证混为一谈。
- 编译源与操作见 [kernels/README.md](kernels/README.md)。P/D ABI 不同，不能混用 `.so`；新构建必须重新验证，不能绕过 hash 校验。

## 复测方法和结果

输入 **81920**，命中 **73728**，新增 prefill **8192**，输出 **600**，`ignore_eos=true`、temperature=0。使用四组独立前缀 seed `221450000..221450003`，每轮尾部 seed 不同。文本经过 tokenizer decode/encode 回环验证；不是把 80K token IDs 解码成文本后仍假定长度不变。

先做四组冷→热放置验证，再每轮 12 条 4-token 输出预热。确认每个 P 都接收对应前缀且再次到达命中；P 日志必须逐条显示 8192 fresh / 73728 cached。API hit 可以来自 D，不足以证明 P 没有重算。所有 CPU prompt 构建完毕、graph/kernel 预热完毕、profiler 退出后才启动计时。

开环固定到达 **3.40 req/s**，每轮 **2072** 请求；预先固定 **120–540 秒**观察窗。流必须收到 `[DONE]` 和完整 usage，以 client terminal 时间计完成，失败不丢弃；总分母包含全部 **40 张 P+D GPU**。同时核验全局及四个前缀组的 inflight/before-first 队列与 TTFT 趋势，而不是只看 offered rate。

| 独立测试 | 保守总 TPS/GPU | 整体 fresh input TPS | 整体 output TPS | TTFT P50/P99 (s) | TPOT P50/P99 (ms) | 完成/错误 |
|---|---:|---:|---:|---|---|---|
| R65 | 7014.18 | 27852.80 | 2040.00 | 2.584 / 3.202 | 33.04 / 63.58 | 2072 / 0 |
| R66 | 7009.29 | 27833.30 | 2038.57 | 2.586 / 4.716 | 33.87 / 67.26 | 2072 / 0 |

聚合证据见 [results/qualification.json](results/qualification.json)，保留原始结果 SHA；没有公开原始请求内容。两个测试运行版本一致，队列检查通过；每个 P 每轮 518 个实测热请求。

**边界：**这是含缓存输入的经济总吞吐，不是纯计算 FLOPS 提升。两轮到达各约 10 分钟、固定观察窗各 7 分钟，不是多小时 SLA。R66 前后半窗约 7082.97 / 6935.61，7K 余量很薄；随机 ignore-EOS 流存在约 20 秒的最大有效文本间隙，不能宣称自然语言流畅度 SLA。真实 MTP不等于通用准确率已验收。四个可复用前缀不等于每请求独立 80K 前缀容量。旧 5K 实验的 tokenizer/前缀口径不同，不能直接算成纯 kernel +40%。

## 未带入的候选及后续验收

MoE tactic 热微测收益未在冷/服务内成立、持久 workspace 退化、FI 0.7 MLA 实际张量正确性未通过、FP8 wire/AR 替代退化，均未作为优化提交。InputSP/完整 KV 的单项百分比不能相加推导最终总收益。

本次整理只做本地源码/CPU 验证，不启动新压测、不改现有服务。统一源码版本或正式镜像另起变更，需重复冷/热前缀、真实层输出、跨 rank Top-K、流完成/取消、L3 restore、graph、显存及两轮开环验收。回退应停流排空后恢复上一套完整镜像与 overlays，不能混搭或仅反向应用其中一个 patch。
