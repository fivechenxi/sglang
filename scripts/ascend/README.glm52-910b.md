# GLM-5.2 W8A8 on Ascend 910B

This branch reproduces the four-node production baseline from SGLang commit
`4ad418d2c3d43cb3c699bc9419d32673b1fca7d8`.

Validated package set:

- SGLang `0.5.15.post2.dev419+g4ad418d2c`;
- `deep_ep 1.0.0+9765e275.cann.9.0.0.b250`;
- `sgl_kernel_npu 2026.6.1`;
- `torch_npu 2.10.0`;
- CANN 9.0.0.

Before launching SGLang in the validated Ascend image, run:

```bash
python3 scripts/ascend/patch_glm52_910b_w8a8_runtime.py
```

Then select the hybrid transport:

```bash
export DEEP_USE_MODE=hybrid
```

The resulting transport split is:

- prefill/extend: normal `ALLTOALL`, with the equal-split fallback implemented
  by `torch.distributed.all_to_all_single`;
- decode: native DeepEP low-latency dispatch/combine;
- decode batch sizes 4, 8, and 16 may be captured by the NPU graph backend.

This is an exact production-recovery branch, not evidence that a newer SGLang
`main` commit works with the package set above. Move the branch forward only
after the same four-node startup and protocol regression passes.
