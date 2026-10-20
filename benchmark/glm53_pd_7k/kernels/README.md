# Exact Top-K 源码与构建边界

`p_topk.cu` 为 SGLang PR #37893 固定源码，原始 TileLang 改编说明保留。`d_topk.cu` 在同一来源上将 radix key 改为 score 高 32 位 + inverse index 低 32 位，解决相同分值跨 rank 选择集合不一致；不是新的低精度近似。

| 角色 | Torch ABI | CUDA / arch | 注册名 | 历史实测库 SHA256 |
|---|---|---|---|---|
| P | 2.13.0+cu130 | 13.0 / 12.0 | glm53_topk_r32 | 953e958cc3a1276b8dc575081a37fb4613f4b9545ac9fd275ccc895b120a5e8a |
| D | 2.11.0+cu130 | 13.0 / 12.0 | glm53_topk_r37 | 14f2430855e54c8fc19a80e323f5fe1dcb0929c97b2f8d30ef0db94b0a438e4d |

在隔离编译目录使用 `torch.utils.cpp_extension.load`，不要覆盖安装包：

```python
# 在对应角色的固定 ABI 镜像中运行；示例 role='P'，D 单独构建。
import os
from pathlib import Path
from torch.utils.cpp_extension import load
import torch

role = 'P'
expected = {'P': '2.13.0+cu130', 'D': '2.11.0+cu130'}[role]
if torch.__version__ != expected:
    raise RuntimeError('Wrong torch ABI')
root = Path('benchmark/glm53_pd_7k/kernels').resolve()
build = root / 'build' / role
build.mkdir(parents=True, exist_ok=False)
os.environ['TORCH_CUDA_ARCH_LIST'] = '12.0'
os.environ['MAX_JOBS'] = '2'
revision = 'r32' if role == 'P' else 'r37'
library = load(
    name='glm53_topk_pr37893_r32' if role == 'P' else 'glm53_topk_deterministic_r37',
    sources=[str(root / (role.lower()+'_topk.cu')), str(root / (role.lower()+'_bindings.cpp'))],
    extra_cflags=['-O3'],
    extra_cuda_cflags=['-O3', '-std=c++17', '-lineinfo',
        '-Dfast_topk_interface=glm53_'+revision+'_topk',
        '-Dfast_topk_transform_interface=glm53_'+revision+'_paged',
        '-Dfast_topk_transform_ragged_interface=glm53_'+revision+'_ragged'],
    with_cuda=True, build_directory=str(build), is_python_module=False, verbose=True)
print(library)
```

这段是从实测 build flags 整理的可移植构建说明；本地此次未重新 CUDA 编译。编译器/绝对构建路径不同可能导致新库 SHA 不同，不承诺二进制可重现。**不能为了让 hook 加载而盲改 `LIB_SHA`**：先检查源码 SHA，并完成实际 score tensor 的精确 oracle、tie、zero-stride、eager/graph、多 rank 和服务端验证，单独记录新库 SHA，再审阅更新绑定。

原 hook 的 P 库位置为 `/research/glm53_topk_pr37893_r32.so`，D 为 `/research-d-exact-topk/glm53_topk_deterministic_r37.so`；均为通用隔离容器挂载点。P/D `.so` 不能混用；本 PR 不包含任何二进制。完整服务准确率和自然输出回归仍是新构建的发布前提。
