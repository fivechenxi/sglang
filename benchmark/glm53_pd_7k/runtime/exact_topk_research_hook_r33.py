"""Default-off, isolated research binding for reviewed PR37893 AOT kernels.

Do not patch a running interpreter or install globally. It preserves the
legacy top-k interfaces; the candidate buffer becomes a cache, not a cutoff.
"""
import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path('/research')
LIB_SHA = '953e958cc3a1276b8dc575081a37fb4613f4b9545ac9fd275ccc895b120a5e8a'
seen = set()


def acknowledge(method):
    if method in seen:
        return
    seen.add(method)
    rank = torch.cuda.current_device()
    path = ROOT / f'exact-topk-r33-r{rank}-{method}.json'
    path.write_text(json.dumps(dict(method=method, rank=rank, library_sha256=LIB_SHA,
                                   community_pr=37893, actual_call=True)))


def paged(score, lengths, page_table_size_1, cu_seqlens_q, topk, row_starts=None):
    if topk != 2048 or score.ndim != 2:
        raise ValueError('Reviewed exact legacy top-k requires matrix and k=2048')
    output = torch.empty((score.shape[0], topk), device=score.device, dtype=torch.int32)
    torch.ops.glm53_topk_r32.paged(score, lengths, output, page_table_size_1, cu_seqlens_q, row_starts)
    acknowledge('paged')
    return output


def ragged(score, lengths, topk_indices_offset, topk, row_starts=None):
    if topk != 2048 or score.ndim != 2:
        raise ValueError('Reviewed exact legacy top-k requires matrix and k=2048')
    output = torch.empty((score.shape[0], topk), device=score.device, dtype=torch.int32)
    torch.ops.glm53_topk_r32.ragged(score, lengths, output, topk_indices_offset, row_starts)
    acknowledge('ragged')
    return output


def install():
    if os.environ.get('GLM53_RESEARCH_EXACT_TOPK') != '1':
        return
    lib = ROOT / 'glm53_topk_pr37893_r32.so'
    if hashlib.sha256(lib.read_bytes()).hexdigest() != LIB_SHA:
        raise RuntimeError('Unreviewed top-k library')
    torch.ops.load_library(str(lib))
    import sgl_kernel
    import sgl_kernel.top_k
    sgl_kernel.fast_topk_transform_fused = paged
    sgl_kernel.fast_topk_transform_ragged_fused = ragged
    sgl_kernel.top_k.fast_topk_transform_fused = paged
    sgl_kernel.top_k.fast_topk_transform_ragged_fused = ragged
    print('GLM53_R33_EXACT_TOPK_INSTALLED pid=' + str(os.getpid()), flush=True)
