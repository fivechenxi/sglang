FROM ubuntu:24.04

ARG MOONCAKE_VERSION=0.3.13.post1
ENV DEBIAN_FRONTEND=noninteractive

ARG MOONCAKE_WHEEL_URL=https://github.com/kvcache-ai/Mooncake/releases/download/v0.3.13.post1/mooncake_transfer_engine_npu-0.3.13.post1-cp312-cp312-manylinux_2_35_aarch64.whl
ARG MOONCAKE_WHEEL_SHA256=ba134f2cc99784aa32404c3a1406e3c85400792fb2c62bb458cd4757a07cc4bb
ARG MOONCAKE_WHEEL_FILE=mooncake_transfer_engine_npu-0.3.13.post1-cp312-cp312-manylinux_2_35_aarch64.whl

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl libcurl4t64 libibverbs1 libnuma1 python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv /opt/mooncake && \
    curl -fsSL "${MOONCAKE_WHEEL_URL}" -o "/tmp/${MOONCAKE_WHEEL_FILE}" && \
    echo "${MOONCAKE_WHEEL_SHA256}  /tmp/${MOONCAKE_WHEEL_FILE}" | sha256sum -c - && \
    /opt/mooncake/bin/pip install --no-cache-dir "/tmp/${MOONCAKE_WHEEL_FILE}" && \
    rm -f "/tmp/${MOONCAKE_WHEEL_FILE}" && \
    ! ldd /opt/mooncake/lib/python3.12/site-packages/mooncake/mooncake_master | grep -q "not found" && \
    test "$(/opt/mooncake/bin/pip show mooncake-transfer-engine-npu | sed -n 's/^Version: //p')" = "${MOONCAKE_VERSION}" && \
    /opt/mooncake/bin/mooncake_master --help 2>&1 | grep -q -- "enable_oplog" && \
    /opt/mooncake/bin/mooncake_master --help 2>&1 | grep -q -- "ha_backend_connstring"

ENV PATH=/opt/mooncake/bin:${PATH}

EXPOSE 50051 8080 9003

ENTRYPOINT ["mooncake_master"]
