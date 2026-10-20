"""Default-off research binding, D torch2.11 ABI. No fusion/layout change."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/research-d-exact-topk')
LIB_SHA = '14f2430855e54c8fc19a80e323f5fe1dcb0929c97b2f8d30ef0db94b0a438e4d'
acknowledged = False


def plain(score, lengths, topk, row_starts=None):
    import torch
    global acknowledged
    if topk != 2048 or score.ndim != 2:
        raise ValueError('Reviewed exact top-k requires matrix and k=2048')
    output = torch.empty((score.shape[0], topk), device=score.device, dtype=torch.int32)
    torch.ops.glm53_topk_r37.plain(score, output, lengths, row_starts)
    if not acknowledged:
        acknowledged = True
        rank = torch.cuda.current_device()
        (ROOT / f'exact-topk-r38-r{rank}.json').write_text(json.dumps(dict(
            actual_call=True, rank=rank, library_sha256=LIB_SHA, plain=True)))
    return output


def install():
    if os.environ.get('GLM53_RESEARCH_D_EXACT_TOPK') != '1':
        return
    import torch
    if torch.__version__ != '2.11.0+cu130':
        raise RuntimeError('D top-k library requires pinned torch ABI')
    lib = ROOT / 'glm53_topk_deterministic_r37.so'
    if hashlib.sha256(lib.read_bytes()).hexdigest() != LIB_SHA:
        raise RuntimeError('D library hash drift')
    torch.ops.load_library(str(lib))
    import sgl_kernel
    import sgl_kernel.top_k
    sgl_kernel.fast_topk_v2 = plain
    sgl_kernel.top_k.fast_topk_v2 = plain
    print('GLM53_R38_D_EXACT_TOPK_INSTALLED pid=' + str(os.getpid()), flush=True)

