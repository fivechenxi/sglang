FROM ghcr.io/fivechenxi/sglang@sha256:82d8fbb6fccb0ad210944a6f9ce836db54d56e5b3e531683ea74602b133147ad

ARG MOONCAKE_VERSION=0.3.13.post1
ARG MOONCAKE_WHEEL_URL=https://files.pythonhosted.org/packages/d6/ed/0e8d4286bead87dd7f353a3e88a3bb24713f6e65985f7a0ed4b2eda8433a/mooncake_transfer_engine_npu-0.3.13.post1-cp311-cp311-manylinux_2_35_aarch64.whl
ARG MOONCAKE_WHEEL_SHA256=0e618bd3a17554ddbea14cb7e0cc0d2e5e63a8e1bde34e49cab7fbd8c2367153

# Keep the verified SGLang, kernel, MemFabric, and model-serving payload intact;
# only align the Mooncake Store client with the HA master protocol version.
RUN python3 -m pip uninstall -y mooncake-transfer-engine-npu && \
    python3 -m pip install --no-cache-dir --no-deps \
      "${MOONCAKE_WHEEL_URL}#sha256=${MOONCAKE_WHEEL_SHA256}" && \
    python3 -c "from importlib.metadata import distribution; d = distribution('mooncake-transfer-engine-npu'); files = tuple(map(str, d.files or ())); assert d.version == '${MOONCAKE_VERSION}'; assert any(f.startswith('mooncake/store') for f in files); assert any(f.startswith('mooncake/engine') for f in files); print(d.metadata['Name'], d.version)" && \
    test "$(python3 -m pip list --format=freeze | grep -ci '^mooncake-transfer-engine-npu==')" = "1"

LABEL org.opencontainers.image.base.name="ghcr.io/fivechenxi/sglang@sha256:82d8fbb6fccb0ad210944a6f9ce836db54d56e5b3e531683ea74602b133147ad" \
      io.maas.change="upgrade-mooncake-npu-client-to-0.3.13.post1-for-ha-master-protocol"

