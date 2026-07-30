FROM quay.io/ascend/sglang@sha256:175e295542c46204cdfcf0d8506d0c45a6c4ad90d7381924dab1f36ef25642de

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

RUN python3 /sgl-workspace/sglang/scripts/ascend/patch_glm52_910b_w8a8_runtime.py

LABEL org.opencontainers.image.revision="$SGLANG_COMMIT" \
      org.opencontainers.image.source="$SGLANG_REPOSITORY" \
      org.opencontainers.image.url="$BUILD_WORKFLOW_URL" \
      io.maas.base-image="quay.io/ascend/sglang@sha256:175e295542c46204cdfcf0d8506d0c45a6c4ad90d7381924dab1f36ef25642de"

CMD ["/bin/bash"]
