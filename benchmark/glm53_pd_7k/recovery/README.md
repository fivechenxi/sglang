# 精度评测期间的回收与路由修复

本目录补充 2026-10-18/19 已实测的修复，仍为**固定旧运行版本的补丁包**，不直接修改当前分支的核心源码。原 7K 性能结果不重写；这些验证不代表新版本精度或性能已完成验收。GPQA 长循环仍在定位，未纳入“已修复”。

## 修复范围

| 补丁/配置 | 问题与处理 |
|---|---|
| `dcp-cpu-retraction.patch` | DCP 的逻辑 token 索引不能直接访问分片 MLA 物理池；CPU offload/restore 按 rank 映射。DSA index key/scale 仍使用原全局索引。 |
| `draft-kv-retraction.patch` | target 恢复而独立 MTP draft KV 丢失；按请求同时保存/恢复所有独立 draft 池，校验池数量，恢复后释放快照。 |
| `overlap-retraction-boundary.patch` | overlap 中 CPU 输出历史可能落后于 GPU KV；仅在即将发生 PD retraction 时排空未处理结果，再检查内存和完成状态。普通 decode 保留 overlap。 |
| `router-prefill-error-propagation.patch` | P 已返回 HTTP 200 但流内报错，D 仍等待 KV；P 内部请求改为非流式，完整校验 P body，将 HTTP/传输/应用层错误联动到 D 取消。D 的客户端流式行为不变。 |
| `accuracy-routing.json` | 默认失衡阈值 64 使 C32 共享前缀锁在一个 DP；使用现有阈值 0/1.0，负载不同时优先均衡，相同时保留 affinity。不是新的调度算法。 |

前三项参考社区 [#39487](https://github.com/sgl-project/sglang/pull/39487)、[#44899](https://github.com/sgl-project/sglang/pull/44899)、[#39743](https://github.com/sgl-project/sglang/pull/39743)，为旧运行版本的适配，不声称原创。DCP 映射前提是当前 allocator 的加宽、对齐逻辑页与有序页内 token；不支持任意无序索引。

## 验证记录

- 实际部署方法 CPU/CUDA 回归 **6/6**：DCP1/2/4/8 所有 rank、重排页/尾段、旧代码越界/错行负对照、target + 两个 draft 池 poison/relocate 后逐字节恢复、独立 index key/scale、overlap 顺序。
- 真实 PD/MTP/DCP **37 次强制 retraction**，四 DP 为 9/10/8/10；**32/32 算术回答正确且 SSE DONE**。关闭强制回收后 4/4 正常流通过。
- 新 Router 使用相同 Cargo.lock 离线编译成功；旧 Router mock 复现 200 SSE 错误后超时和 32 请求集中单 DP。
- 新 Router 8 类 P 故障均及时失败并关闭对应 D 连接；正常 completion/chat、非流式、logprobs 路径和客户端取消通过。32 个 held 请求分配 **8/8/8/8**，最后负载归零。
- 真实模型 **32/32** 流式请求 HTTP 200 + DONE，四 DP 峰值负载 **8/8/8/8**。
- 隔离故障 P + 真实 D：HTTP 500 和 HTTP 200 SSE error 均约 **1.00 秒**返回客户端错误；D 日志确认约 3–4 秒内 `AbortReq`，队列排空。注入只在专用测试路由，完成后停用。

聚合与原始证据哈希见 `validation.json`。不发布机器地址、容器部署信息、用户请求或 GPQA 原题。

**未覆盖：**提前 D 失败对 P 的反向取消、其他协议/其他版本、L3 恢复、完整精度 A/B 和新增修复后的两轮 7K 压测。32×64K 超过总 KV 容量，路由均衡不能解决这种容量超配，也不能证明循环由路由引起。power-of-two 候选在此旧 Router 的物理 D 数量校验阶段失败，未采用。

## 应用与回退

1. 停流排空，保留整套旧镜像、命令、环境与 overlays；只在隔离源码副本应用。不能热改运行中的模块。
2. 先检查本目录 `manifest.json` 的每文件 pre/post SHA。D 的前置状态是主目录已验证的 legacy D overlays；不能直接应用到干净 v0.5.17/v0.5.19 或当前 main。
3. D 按表格前三项顺序应用。Router 固定 base `0bcd822377da7b5718e674eaf9c870d349424dd1`，先应用主目录 `patches/router-stream-load.patch`，再应用本目录 Router patch。使用同一 Cargo.lock 重建，不混用 Python ABI。
4. 执行 `python3 check_sources.py --role D --source-root /isolated/runtime`（Router 改用 `--role Router`）。检查工具只读；未知文件或同一个多文件补丁半应用状态立即拒绝。核对后自行在副本逐项 `git apply`，再检查 post SHA。
5. 回退停流后恢复整套已保留运行环境，不能只回退 allocator 或 Req 一侧。Router 恢复旧库和配置也会恢复旧错误行为。

## 回归脚本

轻量打包检查（仓库根目录）：

```sh
python3 -m unittest discover -s benchmark/glm53_pd_7k/recovery/tests -v
```

真实 D 方法测试必须在上述 patched runtime 的隔离 GPU 容器中执行，不是当前 checkout，也不是在线服务进程内：

```sh
python3 integration/test_retraction.py
```

它导入实际 SGLang 类，使用小 CPU/CUDA 张量；`integration/fixtures/` 仅保留旧方法作负对照，不能用于部署。测试不跳过 CUDA，缺失环境会失败。

Router 测试只用 loopback HTTP mock，不连接模型、无需 GPU；先按固定 base 构建扩展并安装至对应 `bindings/python/src`：

```sh
python3 integration/router_error_regression.py --mode stock \
  --source-root /isolated/old/sgl-model-gateway --output-dir /tmp/router-stock-evidence
python3 integration/router_error_regression.py --mode candidate \
  --source-root /isolated/new/sgl-model-gateway --output-dir /tmp/router-new-evidence
```

需 `aiohttp`、已构建 Router 的 Python 依赖和空闲本地端口 18459/29459/30559/30560。拒绝覆盖已有报告；使用新的输出目录保存每次结果。stock 模式断言旧问题被复现，candidate 模式断言问题消失。
