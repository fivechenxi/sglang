from unittest.mock import Mock, patch

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


def test_batch_transfer_binds_executor_thread_to_engine_npu():
    engine = _build_engine_with_ports([12345])
    engine.npu_id = 7
    parent_batch_transfer = Mock(return_value=0)

    with (
        patch.object(
            ascend_transfer_engine.torch.npu, "current_device", return_value=0
        ),
        patch.object(ascend_transfer_engine.torch.npu, "set_device") as set_device,
        patch.object(
            ascend_transfer_engine.MooncakeTransferEngine,
            "batch_transfer_sync",
            parent_batch_transfer,
        ),
    ):
        ret = engine.batch_transfer_sync("peer:1", [1], [2], [3])

    assert ret == 0
    set_device.assert_called_once_with(7)
    parent_batch_transfer.assert_called_once_with("peer:1", [1], [2], [3])


def test_batch_transfer_does_not_rebind_correct_npu():
    engine = _build_engine_with_ports([12345])
    engine.npu_id = 7

    with (
        patch.object(ascend_transfer_engine.torch.npu, "current_device", return_value=7),
        patch.object(ascend_transfer_engine.torch.npu, "set_device") as set_device,
        patch.object(
            ascend_transfer_engine.MooncakeTransferEngine,
            "batch_transfer_sync",
            return_value=0,
        ),
    ):
        engine.batch_transfer_sync("peer:1", [1], [2], [3])

    set_device.assert_not_called()


def test_send_probe_binds_probe_thread_to_engine_npu():
    engine = _build_engine_with_ports([12345])
    engine.npu_id = 7

    with (
        patch.object(
            ascend_transfer_engine.torch.npu, "current_device", return_value=0
        ),
        patch.object(ascend_transfer_engine.torch.npu, "set_device") as set_device,
        patch.object(
            ascend_transfer_engine.MooncakeTransferEngine,
            "send_probe",
            return_value=0,
        ) as parent_send_probe,
    ):
        ret = engine.send_probe("peer:1")

    assert ret == 0
    set_device.assert_called_once_with(7)
    parent_send_probe.assert_called_once_with("peer:1")
