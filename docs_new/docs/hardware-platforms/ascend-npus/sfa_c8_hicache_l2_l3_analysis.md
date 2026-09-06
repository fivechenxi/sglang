---
title: "SFA C8 HiCache L2/L3 方案分析"
description: "记录 SFA C8 Host 布局、prefix 匹配和 Mooncake 对象边界的设计依据。"
---

# SFA C8 HiCache L2/L3 方案分析

## 问题

原设计把 target SFA、target Indexer、draft SFA 和 draft Indexer 建模为四个
Mooncake 对象，并分别检查命中。需要确认这种拆分是否由数据布局或 Mooncake API
要求。

## 已确认事实

### Prefix 匹配不读取缓存 payload

HiCache 根据 token page 计算 hash，并保留现有 `extra_key`。Radix tree 和 L3
`batch_exists` 匹配的是 key，不比较 SFA、Indexer 或 draft 的缓存字节。

因此四段数据来自同一个 prefix，只需要一次匹配。SFA main 可以拥有 allocator 和
host node，但它仍然只是 payload，不是匹配内容本身。

### Mooncake segment 不定义对象边界

Mooncake 注册 Host segment，只是允许 Transfer Engine 访问一段地址：

```text
register(base_address, size)
```

注册操作不知道这段地址属于 SFA、Indexer 或 draft，也不决定一个 storage key
对应几个 segment。

### Mooncake 支持一个对象对应多个 buffer

当前 backend 已使用 `batch_put_from_multi_buffers` 和
`batch_get_into_multi_buffers`。一个 key 可以引用多个已注册的 pointer/size，适合
直接表达一个跨四个 Host region 的 logical page。

### 后三段是必需 payload，不是独立缓存

Lightning Indexer 和 NEXTN draft 状态需要随命中页恢复，否则命中后的 SFA 或
speculative decode 状态不完整。但它们不改变 prefix 是否匹配，只影响该页能否被
标记为 valid。

## 方案比较

### 四个对象

```text
page_hash_main
page_hash_indexer
page_hash_draft
page_hash_draft_indexer
```

该方案可以复用现有 `PoolTransfer` sidecar，但会引入四次存在性判断、部分写成功、
写入顺序、orphan 清理和命中长度求交。这些复杂度来自对象拆分，不是 SFA C8 的
要求。

### 一个连续大 buffer

可以把四段数据打包到一块连续 Host 内存，再用一个 key 存储。对象简单，但每次
写入和读取都需要额外 pack/unpack，或设计新的统一物理布局。

### 一个对象加多个 buffer

```text
page_hash -> [target SFA, target LI, draft SFA, draft LI]
```

Host 继续使用原生 dtype 和布局；Mooncake 通过 multi-buffer API 对一个 key 读写
多个 page slice。该方案没有临时拼接，也没有多对象一致性问题。

## 决定

采用“一个逻辑页、一个 key、四段 payload”：

- L2 使用一个 allocator 和一个 valid 状态；
- 四段 Host region 共用 logical page ID；
- prefix lookup 只执行一次；
- L3 每页只创建一个 Mooncake object；
- put/get 使用 multi-buffer zero-copy；
- 任一段失败时整页无效并回退 recompute。

不采用四个 `PoolName` sidecar 作为独立 L3 对象，也不对四段 payload 分别做
prefix match。

## 实现前实测项

1. 当前 Mooncake 版本的 multi-buffer API 对每个 key 返回单一、完整的结果。
2. 四个 page-first buffer 的地址、大小、对齐和注册范围正确。
3. P 侧 NEXTN draft 的真实层数、dtype 和 layer mapping 与 descriptor 一致。
4. 不同 attention rank 的 payload 是否一致；不一致时 L3 key 必须包含 rank。
5. 四 buffer iovec 与单连续 buffer 的 L3 带宽和 P90 latency。

这些项目影响实现细节和性能，不改变单 key、单匹配的逻辑模型。
