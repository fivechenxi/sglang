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
    python3 -m pip show memfabric-hybrid | grep -q '^Version: 1.2.0$'

# HiCache L3 uses Mooncake Store independently of the Ascend MemFabric P/D
# transport. Pin the HA-compatible NPU CPython 3.11 ARM64 wheel and verify both the Store and
# Transfer Engine payload at image-build time. The native extension itself is
# imported by the in-cluster smoke test because libascend_hal.so is supplied by
# the host driver mount, not by the portable image build environment.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libibverbs1 && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m pip install --no-cache-dir \
    "https://files.pythonhosted.org/packages/d6/ed/0e8d4286bead87dd7f353a3e88a3bb24713f6e65985f7a0ed4b2eda8433a/mooncake_transfer_engine_npu-0.3.13.post1-cp311-cp311-manylinux_2_35_aarch64.whl#sha256=0e618bd3a17554ddbea14cb7e0cc0d2e5e63a8e1bde34e49cab7fbd8c2367153" && \
    python3 -c "from importlib.metadata import distribution; d = distribution('mooncake-transfer-engine-npu'); files = tuple(map(str, d.files or ())); assert d.version == '0.3.13.post1'; assert any(f.startswith('mooncake/store') for f in files); assert any(f.startswith('mooncake/engine') for f in files); print(d.metadata['Name'], d.version)"

# MemFabric loads HCOM again by its bare filename when it creates a lazy
# device-RDMA connection. The wheel preloads the absolute file, but its SONAME
# is libhcom.so.0, so that preload does not satisfy a later dlopen("libhcom.so").
# Keep the wheel's private lib directory in the process startup search path.
ENV LD_LIBRARY_PATH=/usr/local/python3.11.15/lib/python3.11/site-packages/memfabric_hybrid/lib:${LD_LIBRARY_PATH}
RUN python3 -c "import ctypes; ctypes.CDLL('libhcom.so'); print('libhcom.so load verified')"

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

# Import the request protocol during the image build. compileall only checks
# syntax and does not execute io_struct's BaseReq naming/protocol validation.
RUN python3 -c "from sglang.srt.managers.io_struct import PrefillAdmissionAckReq; print(PrefillAdmissionAckReq.__name__)"

# W4A8C8 uses the same verified EP32/DP8/TP4 DeepEP runtime policy as W8A8.
RUN python3 /sgl-workspace/sglang/scripts/ascend/patch_glm52_910b_w8a8_runtime.py

LABEL org.opencontainers.image.revision="$SGLANG_COMMIT" \
      org.opencontainers.image.source="$SGLANG_REPOSITORY" \
      org.opencontainers.image.url="$BUILD_WORKFLOW_URL" \
      io.maas.model-quantization="w4a8c8" \
      io.maas.base-image="ghcr.io/fivechenxi/sglang@sha256:1e3fa2b90bd184dbf2e73cdef1fc5453d5edc5849aadbbde126f93d06082e43e"

CMD ["/bin/bash"]
