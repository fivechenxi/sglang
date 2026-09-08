FROM ubuntu:24.04

ARG MOONCAKE_VERSION=0.3.11
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates libcurl4t64 libibverbs1 libnuma1 python3 python3-pip python3-venv && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv /opt/mooncake && \
    /opt/mooncake/bin/pip install --no-cache-dir \
      "https://files.pythonhosted.org/packages/08/6f/e8d4307c63cf88bf84e8e62ea6540487e18c1c5530faac0a8d6db0e2894b/mooncake_transfer_engine-0.3.11-cp312-cp312-manylinux_2_39_aarch64.whl#sha256=72a52df441ed2f88d63e3c1e58f2f48551ee1778c07b386bf3be78e6ce6fa1e1" && \
    /opt/mooncake/bin/mooncake_master --help >/dev/null && \
    test "$(/opt/mooncake/bin/pip show mooncake-transfer-engine | sed -n 's/^Version: //p')" = "${MOONCAKE_VERSION}"

ENV PATH=/opt/mooncake/bin:${PATH}

EXPOSE 50051 8080 9003

ENTRYPOINT ["mooncake_master"]
