FROM ghcr.io/fivechenxi/sglang@sha256:1e3fa2b90bd184dbf2e73cdef1fc5453d5edc5849aadbbde126f93d06082e43e

ARG SGLANG_REPOSITORY
ARG SGLANG_COMMIT
ARG BUILD_WORKFLOW_URL=""

# MemFabric 1.0.8 builds a fixed transfer world and cannot safely recover when
# P/D ranks restart independently. 1.2.0 adds dynamic join, background HCOM
# reconnect, and the A2/PD reliability fixes required by this deployment.
# Pin the official ARM64 wheel by digest so the runtime remains reproducible.
RUN python3 -m pip install --no-cache-dir --upgrade \
    "https://files.pythonhosted.org/packages/d4/6b/750a2f834e4f3bc9d2066d534ba53468ff7ed49c85d0c44802c79d2d45b4/memfabric_hybrid-1.2.0-cp311-cp311-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl#sha256=5213be7e6384923612447d79828dd6c45500525449f0849a200337a7cf20f442" && \
    python3 -c "import memfabric_hybrid; print(memfabric_hybrid.__file__)"

RUN test -n "$SGLANG_REPOSITORY" && test -n "$SGLANG_COMMIT" && \
    rm -rf /sgl-workspace/sglang && \
    git clone "$SGLANG_REPOSITORY" /sgl-workspace/sglang && \
    cd /sgl-workspace/sglang && \
    git checkout --detach "$SGLANG_COMMIT" && \
    test "$(git rev-parse HEAD)" = "$SGLANG_COMMIT" && \
    cd python && \
    rm -f pyproject.toml && \
    mv pyproject_npu.toml pyproject.toml && \
    python3 -m pip install --no-cache-dir --no-deps -e .

# W4A8C8 uses the same verified EP32/DP8/TP4 DeepEP runtime policy as W8A8.
RUN python3 /sgl-workspace/sglang/scripts/ascend/patch_glm52_910b_w8a8_runtime.py

LABEL org.opencontainers.image.revision="$SGLANG_COMMIT" \
      org.opencontainers.image.source="$SGLANG_REPOSITORY" \
      org.opencontainers.image.url="$BUILD_WORKFLOW_URL" \
      io.maas.model-quantization="w4a8c8" \
      io.maas.base-image="ghcr.io/fivechenxi/sglang@sha256:1e3fa2b90bd184dbf2e73cdef1fc5453d5edc5849aadbbde126f93d06082e43e"

CMD ["/bin/bash"]
