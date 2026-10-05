import unittest

from utils import make_serving  # noqa: F401 — bootstrap import

from sglang.srt.entrypoints.openai.protocol import ResponsesRequest, UsageInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class ResponsesRequestTestCase(unittest.TestCase):
    def test_pd_internal_fields_are_preserved(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            stream=True,
            bootstrap_host="prefill.internal",
            bootstrap_port=8998,
            bootstrap_room=42,
            pd_prefill_admission_ack=True,
            decode_token_reservation_id="lease-1",
            routed_dp_rank=3,
            disagg_prefill_dp_rank=2,
            store=False,
        )

        self.assertEqual(request.bootstrap_room, 42)
        self.assertTrue(request.pd_prefill_admission_ack)
        self.assertEqual(request.decode_token_reservation_id, "lease-1")
        self.assertEqual(request.routed_dp_rank, 3)
        self.assertEqual(request.disagg_prefill_dp_rank, 2)

    def test_legacy_dp_rank_wire_field_is_normalized(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            data_parallel_rank=3,
            store=False,
        )
        self.assertEqual(request.data_parallel_rank, 3)
        self.assertEqual(request.routed_dp_rank, 3)

        explicit = ResponsesRequest(
            model="x",
            input="hi",
            data_parallel_rank=3,
            routed_dp_rank=1,
            store=False,
        )
        self.assertEqual(explicit.routed_dp_rank, 1)

    def test_function_tool_accepted(self):
        request = ResponsesRequest(
            model="x",
            input="call the tool",
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string"}},
                        "required": ["key"],
                    },
                    "strict": True,
                }
            ],
            store=False,
        )

        self.assertEqual(request.tools[0].type, "function")
        self.assertEqual(request.tools[0].name, "lookup")
        self.assertTrue(request.tools[0].strict)

    def test_function_tool_requires_name(self):
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x", input="hi", tools=[{"type": "function"}], store=False
            )
        with self.assertRaises(ValueError):
            ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": "function", "name": ""}],
                store=False,
            )

    def test_extended_tool_types_accepted(self):
        for tool_type in (
            "web_search",
            "web_search_preview",
            "code_interpreter",
            "file_search",
            "image_generation",
            "computer_use_preview",
            "local_shell",
            "mcp",
            "custom",
            "namespace",
        ):
            request = ResponsesRequest(
                model="x",
                input="hi",
                tools=[{"type": tool_type}],
                store=False,
            )
            self.assertEqual(request.tools[0].type, tool_type)

    def test_namespace_tool_carries_inner_tools_list(self):
        request = ResponsesRequest(
            model="x",
            input="hi",
            tools=[
                {
                    "type": "namespace",
                    "name": "codex",
                    "tools": [
                        {"type": "function", "name": "apply_patch"},
                        {"type": "function", "name": "shell"},
                    ],
                }
            ],
            store=False,
        )
        self.assertEqual(request.tools[0].type, "namespace")
        self.assertEqual(len(request.tools[0].tools), 2)
        self.assertEqual(request.tools[0].tools[0]["name"], "apply_patch")


class ResponsesSamplingParamsTestCase(unittest.TestCase):
    def test_processed_stop_and_tool_constraint_propagate(self):
        request = ResponsesRequest(model="x", input="call the tool", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            stop=["</s>"],
            tool_call_constraint=("json_schema", {"type": "object"}),
        )
        self.assertEqual(params["stop"], ["</s>"])
        self.assertEqual(params["json_schema"], '{"type": "object"}')

    def test_constraint_conflict_raises(self):
        request = ResponsesRequest(model="x", input="hi", store=False)
        with self.assertRaises(ValueError):
            request.to_sampling_params(
                default_max_tokens=128,
                default_params={"json_schema": '{"type": "object"}'},
                tool_call_constraint=("json_schema", {"type": "object"}),
            )

    def test_structural_tag_with_model_dump(self):
        class _FakeStructuralTag:
            def model_dump(self, by_alias=False):
                return {"type": "structural_tag"}

        request = ResponsesRequest(model="x", input="hi", store=False)
        params = request.to_sampling_params(
            default_max_tokens=128,
            default_params={},
            tool_call_constraint=("structural_tag", _FakeStructuralTag()),
        )
        self.assertEqual(params["structural_tag"], '{"type": "structural_tag"}')


class ResponsesResponseFromRequestTestCase(unittest.TestCase):
    def test_parallel_tool_calls_false_preserved(self):
        from sglang.srt.entrypoints.openai.protocol import ResponsesResponse

        request = ResponsesRequest(
            model="x", input="hi", parallel_tool_calls=False, store=False
        )
        response = ResponsesResponse.from_request(
            request,
            sampling_params={},
            model_name="x",
            created_time=0,
            output=[],
            status="completed",
            usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )
        self.assertFalse(response.parallel_tool_calls)


if __name__ == "__main__":
    unittest.main()
