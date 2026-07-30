FROM ghcr.io/fivechenxi/sglang@sha256:1e3fa2b90bd184dbf2e73cdef1fc5453d5edc5849aadbbde126f93d06082e43e

ARG SGLANG_REPOSITORY
ARG SGLANG_COMMIT
ARG BUILD_WORKFLOW_URL=""

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
