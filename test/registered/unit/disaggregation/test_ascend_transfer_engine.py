from unittest.mock import patch

from sglang.srt.disaggregation.ascend import transfer_engine as ascend_transfer_engine
from sglang.srt.disaggregation.utils import DisaggregationMode


class _FakeTransferEngine:
    class TransDataOpType:
        SDMA = object()
        DEVICE_RDMA = object()

    def __init__(self, ports):
        self._ports = iter(ports)

    def get_rpc_port(self):
        return next(self._ports)


def _build_engine_with_ports(ports):
    fake_engine = _FakeTransferEngine(ports)
    with (
        patch.object(ascend_transfer_engine, "import_error", None),
        patch.object(
            ascend_transfer_engine, "TransferEngine", return_value=fake_engine
        ),
        patch.object(
            ascend_transfer_engine.AscendTransferEngine,
            "initialize",
            autospec=True,
        ),
    ):
        engine = ascend_transfer_engine.AscendTransferEngine(
            hostname="10.0.0.7",
            npu_id=0,
            disaggregation_mode=DisaggregationMode.PREFILL,
        )
    return engine


def test_session_id_refreshes_when_rpc_port_is_assigned_during_initialize():
    engine = _build_engine_with_ports([0, 23456])

    assert engine.session_id == "10.0.0.7:23456"


def test_session_id_keeps_preallocated_rpc_port():
    engine = _build_engine_with_ports([12345])

    assert engine.session_id == "10.0.0.7:12345"
