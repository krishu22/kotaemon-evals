import asyncio
import time
from types import SimpleNamespace

import httpx
from openai import AsyncOpenAI
from openai import OpenAI
from pydantic import BaseModel

from evals.operations import (
    OperationsRecorder,
    _instrument_method,
    build_batch_operations_summary,
    instrument_openai_calls,
)


class _StructuredOutput(BaseModel):
    answer: str


def _mock_completion_response():
    return {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
    }


def test_evaluation_delay_is_outside_request_latency():
    recorder = OperationsRecorder()

    recorder.start_request()
    time.sleep(0.01)
    recorder.start_retrieval()
    recorder.first_visible_token()
    time.sleep(0.01)
    recorder.finish_request(True)
    request_latency = recorder.snapshot()["timings_ms"]["application_request"]

    recorder.start_evaluation()
    time.sleep(0.05)
    recorder.finish_evaluation()
    recorder.finish_request(False, "evaluation")
    operations = recorder.snapshot()

    assert operations["timings_ms"]["application_request"] == request_latency
    assert operations["timings_ms"]["application_ttft"] is not None
    assert operations["timings_ms"]["pipeline"] < request_latency
    assert operations["timings_ms"]["evaluation"] >= 40
    assert operations["requests"] == {"status": "complete", "failed_stage": None}


def test_stream_instrumentation_preserves_generator_return_value():
    class FakePipeline:
        def stream(self):
            yield SimpleNamespace(channel="chat", content="answer")
            return SimpleNamespace(text="answer")

    recorder = OperationsRecorder()
    recorder.start_request()
    pipeline = FakePipeline()
    _instrument_method(
        pipeline,
        "stream",
        recorder,
        "application_request",
        "application",
        stream=True,
        request_stream=True,
    )

    stream = pipeline.stream()
    assert next(stream).content == "answer"
    try:
        next(stream)
    except StopIteration as stop:
        assert stop.value.text == "answer"
    else:
        raise AssertionError("stream did not finish")


def test_async_openai_usage_is_recorded_from_provider_response():
    recorder = OperationsRecorder()

    async def handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=_mock_completion_response())

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as http_client:
            client = AsyncOpenAI(api_key="test-key", http_client=http_client)
            with instrument_openai_calls(), recorder.activate("evaluation"):
                response = await client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "test"}],
                    response_format=_StructuredOutput,
                )
            assert response.choices[0].message.parsed.answer == "ok"
            await client.close()

    asyncio.run(run())
    calls = recorder.snapshot()["model_calls"]

    assert len(calls) == 1
    assert calls[0]["component"] == "evaluator"
    assert calls[0]["transport"] == "async"
    assert calls[0]["usage"]["total_tokens"] == 5


def test_sync_structured_output_usage_is_recorded():
    recorder = OperationsRecorder()

    def handler(request):
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=_mock_completion_response())

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = OpenAI(api_key="test-key", http_client=http_client)
        with instrument_openai_calls(), recorder.activate("evaluation"):
            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "test"}],
                response_format=_StructuredOutput,
            )
        assert response.choices[0].message.parsed.answer == "ok"
        client.close()

    calls = recorder.snapshot()["model_calls"]
    assert len(calls) == 1
    assert calls[0]["component"] == "evaluator"
    assert calls[0]["transport"] == "sync"
    assert calls[0]["usage"]["total_tokens"] == 5


def test_batch_summary_keeps_originating_calls_separate_and_counts_failures():
    results = [
        {
            "status": "complete",
            "operations": {
                "requests": {"status": "complete"},
                "timings_ms": {
                    "application_request": 100,
                    "pipeline": 100,
                    "application_ttft": 20,
                    "generator_ttft": 10,
                },
                "model_calls": [
                    {
                        "component": "evaluator",
                        "usage": {
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                        },
                        "cost": 0.1,
                    }
                ],
                "originating_run": {
                    "requests": {"status": "complete"},
                    "setup": {
                        "components": {
                            "setup": {"model_calls": 1, "tokens": {"total": 7}}
                        }
                    },
                    "model_calls": [
                        {
                            "component": "generator",
                            "usage": {
                                "input_tokens": 5,
                                "output_tokens": 4,
                                "total_tokens": 9,
                            },
                            "cost": 0.2,
                        }
                    ],
                },
            },
        },
        {
            "status": "failed",
            "failed_stage": "kotaemon_run",
            "operations": {
                "requests": {
                    "status": "failed",
                    "failed_stage": "kotaemon_run",
                },
                "timings_ms": {"application_request": 50},
                "model_calls": [],
            },
        },
    ]

    summary = build_batch_operations_summary(results)

    assert summary["request_count"] == 2
    assert summary["request_success_count"] == 1
    assert summary["failures_by_stage"] == {"kotaemon_run": 1}
    assert summary["usage_and_cost"]["generator"]["model_calls"] == 1
    assert summary["usage_and_cost"]["evaluator"]["model_calls"] == 1
    assert summary["usage_and_cost"]["application"]["model_calls"] == 1
    assert summary["usage_and_cost"]["pipeline"]["model_calls"] == 1
    assert (
        summary["setup_accounted_once"]["components"]["setup"]["tokens"]["total"] == 7
    )
