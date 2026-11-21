"""Operational measurements for live Kotaemon and evaluation runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, copy_context
import json
import math
import os
import statistics
import threading
import time
from typing import Any, AsyncIterator, Callable, Generator, Iterator


_ACTIVE_RECORDER: ContextVar[OperationsRecorder | None] = ContextVar(
    "kotaemon_operations_recorder", default=None
)
_ACTIVE_PURPOSE: ContextVar[str] = ContextVar(
    "kotaemon_operations_purpose", default="unclassified"
)
_ACTIVE_HTTP_ATTEMPTS: ContextVar[list[int] | None] = ContextVar(
    "kotaemon_operations_http_attempts", default=None
)


def _value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def _usage(response: Any) -> dict[str, int | None] | None:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return None
    usage = _value(usage)
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    return {
        "input_tokens": prompt if isinstance(prompt, int) else None,
        "output_tokens": completion if isinstance(completion, int) else None,
        "total_tokens": total if isinstance(total, int) else None,
    }


def _pricing() -> tuple[dict[str, Any], str, str]:
    raw = os.getenv("KOTAEMON_EVAL_PRICING_JSON", "").strip()
    currency = os.getenv("KOTAEMON_EVAL_PRICING_CURRENCY", "USD").strip() or "USD"
    if not raw:
        return {}, currency, "not_configured"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("KOTAEMON_EVAL_PRICING_JSON must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("KOTAEMON_EVAL_PRICING_JSON must be a model-to-rates object")
    return parsed, currency, "KOTAEMON_EVAL_PRICING_JSON"


def _yield_stream(
    stream: Any, on_chunk: Callable[[Any], None] | None = None
) -> Generator[Any, None, Any]:
    iterator = iter(stream)
    while True:
        try:
            chunk = next(iterator)
        except StopIteration as stop:
            return stop.value
        if on_chunk is not None:
            on_chunk(chunk)
        yield chunk


class OperationsRecorder:
    """Collect wall-clock timings and leaf provider calls without trace rollup."""

    def __init__(self) -> None:
        self.started_at = time.monotonic()
        self._timings: dict[str, float] = {}
        self._calls: list[dict[str, Any]] = []
        self._call_id = 0
        self._call_lock = threading.Lock()
        self._request_started: float | None = None
        self._retrieval_started: float | None = None
        self._pipeline_latency_ms: float | None = None
        self._request_ttft: float | None = None
        self._generator_started: float | None = None
        self._generator_ttft: float | None = None
        self._request_latency_ms: float | None = None
        self._generator_latency_ms: float | None = None
        self._request_status = "not_started"
        self._failed_stage: str | None = None
        self._evaluation_started: float | None = None
        self._evaluation_duration_ms: float | None = None
        self._serialization_ms: float | None = None
        self._trace_upload_wait_ms: float | None = None
        self._trace_usage_fetch_ms: float | None = None
        self._langsmith_usage: dict[str, Any] | None = None
        self.model_metadata: dict[str, Any] = {}

    @contextmanager
    def activate(self, purpose: str) -> Generator[None, None, None]:
        recorder_token = _ACTIVE_RECORDER.set(self)
        purpose_token = _ACTIVE_PURPOSE.set(purpose)
        try:
            yield
        finally:
            _ACTIVE_PURPOSE.reset(purpose_token)
            _ACTIVE_RECORDER.reset(recorder_token)

    @contextmanager
    def timed(self, name: str) -> Generator[None, None, None]:
        start = time.monotonic()
        try:
            yield
        finally:
            self.add_timing(name, time.monotonic() - start)

    def add_timing(self, name: str, seconds: float) -> None:
        self._timings[name] = self._timings.get(name, 0.0) + seconds * 1000

    def start_request(self) -> None:
        if self._request_started is None:
            self._request_started = time.monotonic()
            self._request_status = "running"

    def first_visible_token(self) -> None:
        now = time.monotonic()
        if self._request_started is not None and self._request_ttft is None:
            self._request_ttft = (now - self._request_started) * 1000
        if self._generator_started is not None and self._generator_ttft is None:
            self._generator_ttft = (now - self._generator_started) * 1000

    def start_retrieval(self) -> None:
        if self._retrieval_started is None:
            self._retrieval_started = time.monotonic()

    def finish_request(self, success: bool, failed_stage: str | None = None) -> None:
        if self._request_started is not None and self._request_latency_ms is None:
            self._request_latency_ms = (time.monotonic() - self._request_started) * 1000
        if self._retrieval_started is not None and self._pipeline_latency_ms is None:
            self._pipeline_latency_ms = (
                time.monotonic() - self._retrieval_started
            ) * 1000
        if self._request_status == "complete":
            return
        self._request_status = "complete" if success else "failed"
        self._failed_stage = None if success else failed_stage

    def start_evaluation(self) -> None:
        if self._evaluation_started is None:
            self._evaluation_started = time.monotonic()

    def finish_evaluation(self) -> None:
        if self._evaluation_started is not None:
            elapsed = (time.monotonic() - self._evaluation_started) * 1000
            self._evaluation_duration_ms = (
                self._evaluation_duration_ms or 0.0
            ) + elapsed
            self._evaluation_started = None

    def record_trace_wait(self, seconds: float) -> None:
        self._trace_upload_wait_ms = seconds * 1000

    def record_trace_usage(self, summary: dict[str, Any], seconds: float) -> None:
        self._langsmith_usage = summary
        self._trace_usage_fetch_ms = seconds * 1000

    def measure_serialization(self, value: dict[str, Any]) -> None:
        start = time.monotonic()
        json.dumps(value, ensure_ascii=False)
        self._serialization_ms = (time.monotonic() - start) * 1000
        value["operations"] = self.snapshot()

    def record_model_call(
        self,
        *,
        endpoint: str,
        model: str | None,
        call_id: int,
        started: float,
        usage: dict[str, int | None] | None,
        error: Exception | None = None,
        external_cost: float | None = None,
        purpose: str | None = None,
        http_attempts: int | None = None,
        transport: str = "sync",
    ) -> None:
        purpose = purpose or _ACTIVE_PURPOSE.get()
        component = {
            "indexing": "setup",
            "query_embedding": "retriever",
            "search": "retriever",
            "reranking": "retriever",
            "evidence_preparation": "retriever",
            "citation_processing": "generator",
            "answer_generation": "generator",
            "evaluation": "evaluator",
        }.get(purpose, "application")
        pricing, currency, pricing_source = _pricing()
        rates = pricing.get(model or "", {})
        cost = None
        if isinstance(rates, dict) and usage is not None:
            input_rate = rates.get(
                f"input_{currency.casefold()}_per_1m",
                rates.get("input_per_1m"),
            )
            output_rate = rates.get(
                f"output_{currency.casefold()}_per_1m",
                rates.get("output_per_1m"),
            )
            input_tokens = usage.get("input_tokens")
            output_tokens = usage.get("output_tokens")
            if endpoint.endswith("embeddings") and (
                isinstance(input_rate, (int, float)) and isinstance(input_tokens, int)
            ):
                cost = input_tokens * input_rate / 1_000_000
            elif (
                isinstance(input_rate, (int, float))
                and isinstance(output_rate, (int, float))
                and isinstance(input_tokens, int)
                and isinstance(output_tokens, int)
            ):
                cost = (
                    input_tokens * input_rate + output_tokens * output_rate
                ) / 1_000_000
        with self._call_lock:
            self._call_id += 1
            unique_call_id = self._call_id
            self._calls.append(
                {
                    "call_id": unique_call_id,
                    "provider_call_id": call_id,
                    "component": component,
                    "purpose": purpose,
                    "provider": endpoint.split(".", 1)[0],
                    "endpoint": endpoint,
                    "transport": transport,
                    "model": model,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    "usage": usage,
                    "usage_source": "provider" if usage is not None else None,
                    "provider_attempts": http_attempts,
                    "retry_count": (
                        max(0, http_attempts - 1) if http_attempts is not None else None
                    ),
                    "retry_coverage": (
                        f"observed_httpx_{transport}_attempts"
                        if http_attempts is not None
                        else "transport_attempts_unavailable"
                    ),
                    "failed": error is not None,
                    "timed_out": (
                        error is not None
                        and (
                            "timeout" in type(error).__name__.casefold()
                            or "timeout" in str(error).casefold()
                        )
                    ),
                    "error_type": type(error).__name__ if error is not None else None,
                    "cost": cost,
                    "external_cost": external_cost,
                    "currency": currency,
                    "pricing_source": pricing_source,
                    "configured_rate": rates if isinstance(rates, dict) else None,
                }
            )

    def _component_totals(self, component: str) -> dict[str, Any]:
        component_names = (
            {"retriever", "generator", "application"}
            if component in {"application", "pipeline"}
            else {component}
        )
        calls = [call for call in self._calls if call["component"] in component_names]
        usage_calls = [call for call in calls if call["usage"] is not None]
        costs = [call["cost"] for call in calls if call["cost"] is not None]
        external_costs = [
            call["external_cost"] for call in calls if call["external_cost"] is not None
        ]
        retry_values = [
            call["retry_count"]
            for call in calls
            if isinstance(call.get("retry_count"), int)
        ]

        def token_sum(key: str) -> int | None:
            values = [
                call["usage"].get(key)
                for call in usage_calls
                if isinstance(call["usage"].get(key), int)
            ]
            return sum(values) if values else None

        return {
            "model_calls": len(calls),
            "failures": sum(call["failed"] for call in calls),
            "timeouts": sum(call["timed_out"] for call in calls),
            "retries": sum(retry_values) if retry_values else None,
            "retries_coverage": (len(retry_values) / len(calls) if calls else None),
            "provider_attempts": sum(
                call["provider_attempts"]
                for call in calls
                if isinstance(call.get("provider_attempts"), int)
            )
            or None,
            "tokens": {
                "input": token_sum("input_tokens"),
                "output": token_sum("output_tokens"),
                "total": token_sum("total_tokens"),
                "provider_usage_calls": len(usage_calls),
                "calls_without_usage": len(calls) - len(usage_calls),
                "coverage": (len(usage_calls) / len(calls) if calls else None),
            },
            "cost": sum(costs) if costs else None,
            "cost_coverage": len(costs) / len(calls) if calls else None,
            "external_reranking_cost": (
                sum(external_costs) if external_costs else None
            ),
            "currency": os.getenv("KOTAEMON_EVAL_PRICING_CURRENCY", "USD"),
        }

    def snapshot(self) -> dict[str, Any]:
        timing = {key: round(value, 3) for key, value in self._timings.items()}
        retrieval_ms = timing.get("retrieval")
        evidence_ms = timing.get("evidence_preparation")
        generation_ms = timing.get("answer_generation")
        request_ms = self._request_latency_ms
        pipeline_ms = self._pipeline_latency_ms
        return {
            "schema_version": 1,
            "timing_definitions": {
                "request_latency_ms": (
                    "Monotonic time from pipeline stream execution start (after "
                    "PDF indexing and runner construction) until the final "
                    "user-facing answer stream is exhausted."
                ),
                "ttft_ms": (
                    "Monotonic time from pipeline stream execution start to the "
                    "first non-empty chat-channel chunk returned to the caller."
                ),
                "generator_ttft_ms": (
                    "Monotonic time from answer-stream start to its first "
                    "non-empty chat-channel chunk."
                ),
                "pipeline_latency_ms": (
                    "Monotonic time from first retrieval-stage entry until the "
                    "final answer stream is exhausted; nested stage durations "
                    "can overlap and must not be summed."
                ),
                "generator_latency_ms": (
                    "Elapsed time while consuming the answer model's stream, "
                    "from provider stream invocation until its final chunk."
                ),
                "model_calls": (
                    "Leaf OpenAI SDK endpoint invocations observed locally; "
                    "LangSmith parent/child spans are not added as extra calls."
                ),
            },
            "metadata": {
                **self.model_metadata,
                "currency": os.getenv("KOTAEMON_EVAL_PRICING_CURRENCY", "USD"),
                "pricing_environment_key": "KOTAEMON_EVAL_PRICING_JSON",
                "langsmith_enabled": bool(
                    self.model_metadata.get("langsmith_trace_id")
                ),
            },
            "timings_ms": {
                **timing,
                "retrieval": retrieval_ms,
                "evidence_preparation": evidence_ms,
                "generator": generation_ms,
                "answer_stream": timing.get("answer_stream"),
                "citation_processing": timing.get("citation_processing"),
                "pipeline": pipeline_ms,
                "application_request": request_ms,
                "application_ttft": (
                    round(self._request_ttft, 3)
                    if self._request_ttft is not None
                    else None
                ),
                "generator_ttft": (
                    round(self._generator_ttft, 3)
                    if self._generator_ttft is not None
                    else None
                ),
                "setup_indexing": timing.get("setup_indexing"),
                "evaluation": self._evaluation_duration_ms,
                "result_serialization": self._serialization_ms,
                "trace_upload_wait": self._trace_upload_wait_ms,
                "langsmith_usage_fetch": self._trace_usage_fetch_ms,
            },
            "requests": {
                "status": self._request_status,
                "failed_stage": self._failed_stage,
            },
            "components": {
                component: self._component_totals(component)
                for component in (
                    "setup",
                    "retriever",
                    "generator",
                    "evaluator",
                    "application",
                    "pipeline",
                )
            },
            "model_calls": list(self._calls),
            "langsmith_trace_usage": self._langsmith_usage,
            "coverage": {
                "retrieval_substage_timing": {
                    "query_embedding_ms": timing.get("query_embedding"),
                    "search_ms": timing.get("search"),
                    "reranking_ms": timing.get("reranking"),
                    "evidence_preparation_ms": evidence_ms,
                },
                "provider_usage": (
                    "actual when returned by the provider; missing per-call values "
                    "remain null and are counted in tokens.calls_without_usage"
                ),
                "cost": (
                    "calculated only when KOTAEMON_EVAL_PRICING_JSON has a model "
                    "rate; otherwise null"
                ),
            },
        }


def build_batch_operations_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize request reliability, latency, usage, and one shared setup."""

    def request_operations(row: dict[str, Any]) -> dict[str, Any]:
        operations = row.get("operations") or {}
        return operations.get("originating_run") or operations

    completed = [
        row
        for row in results
        if request_operations(row).get("requests", {}).get("status") == "complete"
    ]
    attempted = [
        row
        for row in results
        if request_operations(row).get("requests", {}).get("status")
        in {"complete", "failed"}
    ]
    failed_by_stage: dict[str, int] = {}
    for row in results:
        operations = request_operations(row)
        stage = operations.get("requests", {}).get("failed_stage") or row.get(
            "failed_stage"
        )
        if row.get("status") != "complete" and stage:
            failed_by_stage[stage] = failed_by_stage.get(stage, 0) + 1

    def percentile(values: list[float], percentile_value: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        rank = max(0, math.ceil(len(ordered) * percentile_value) - 1)
        return ordered[rank]

    def latency_summary(field: str) -> dict[str, float | int | None]:
        values = []
        for row in results:
            value = request_operations(row).get("timings_ms", {}).get(field)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return {
            "samples": len(values),
            "median_ms": statistics.median(values) if values else None,
            "p95_ms": percentile(values, 0.95),
        }

    def row_model_calls(row: dict[str, Any]) -> list[dict[str, Any]]:
        operations = row.get("operations") or {}
        calls = list(operations.get("model_calls", []))
        originating = operations.get("originating_run")
        if isinstance(originating, dict):
            calls.extend(originating.get("model_calls", []))
        return calls

    component_usage: dict[str, Any] = {}
    component_sources = {
        "retriever": {"retriever"},
        "generator": {"generator"},
        "evaluator": {"evaluator"},
        "application": {"application", "retriever", "generator"},
        "pipeline": {"application", "retriever", "generator"},
    }
    for component, source_components in component_sources.items():
        calls = [
            call
            for row in results
            for call in row_model_calls(row)
            if call.get("component") in source_components
        ]
        token_values = {
            key: [
                call["usage"].get(key)
                for call in calls
                if isinstance(call.get("usage"), dict)
                and isinstance(call["usage"].get(key), int)
            ]
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        costs = [
            call.get("cost")
            for call in calls
            if isinstance(call.get("cost"), (int, float))
        ]
        component_usage[component] = {
            "model_calls": len(calls),
            "usage_calls": sum(call.get("usage") is not None for call in calls),
            "calls_without_usage": sum(call.get("usage") is None for call in calls),
            "token_totals": {
                key: (sum(values) if values else None)
                for key, values in token_values.items()
            },
            "token_averages_per_request": {
                key: (sum(values) / len(results) if values and results else None)
                for key, values in token_values.items()
            },
            "cost_total": sum(costs) if costs else None,
            "cost_average_per_request": (
                sum(costs) / len(results) if costs and results else None
            ),
            "cost_coverage": len(costs) / len(calls) if calls else None,
        }

    setup = next(
        (
            (
                (row.get("operations") or {}).get("setup")
                or request_operations(row).get("setup")
            )
            for row in results
            if (
                (row.get("operations") or {}).get("setup") is not None
                or request_operations(row).get("setup") is not None
            )
        ),
        None,
    )
    trace_usages = [
        usage
        for row in results
        for usage in [
            (row.get("operations") or {}).get("langsmith_trace_usage")
            or request_operations(row).get("langsmith_trace_usage")
        ]
        if isinstance(usage, dict)
    ]
    return {
        "request_count": len(attempted),
        "request_success_count": len(completed),
        "request_failure_count": len(attempted) - len(completed),
        "request_success_rate": (
            len(completed) / len(attempted) if attempted else None
        ),
        "failed_questions": sum(row.get("status") != "complete" for row in results),
        "failures_by_stage": failed_by_stage,
        "latency": {
            "request": latency_summary("application_request"),
            "pipeline": latency_summary("pipeline"),
            "ttft": latency_summary("application_ttft"),
            "generator_ttft": latency_summary("generator_ttft"),
        },
        "usage_and_cost": component_usage,
        "langsmith_trace_usage_non_additive": trace_usages,
        "setup_accounted_once": setup,
        "usage_coverage": {
            "model_calls_with_usage": sum(
                component_usage[name]["usage_calls"]
                for name in ("retriever", "generator", "evaluator", "application")
            ),
            "model_calls_without_usage": sum(
                component_usage[name]["calls_without_usage"]
                for name in ("retriever", "generator", "evaluator", "application")
            ),
        },
    }


def _instrument_method(
    target: Any,
    method_name: str,
    recorder: OperationsRecorder,
    timing_name: str,
    purpose: str,
    *,
    stream: bool = False,
    request_stream: bool = False,
    observe_generator: bool = True,
    external_provider: str | None = None,
) -> None:
    original = getattr(target, method_name, None)
    if original is None or getattr(original, "_operations_wrapped", False):
        return

    if stream:

        def wrapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
            active_recorder = _ACTIVE_RECORDER.get() or recorder
            if request_stream:
                active_recorder.start_request()
            started = time.monotonic()
            if not request_stream and observe_generator:
                active_recorder._generator_started = started
            try:
                with active_recorder.activate(purpose):

                    def observe(chunk: Any) -> None:
                        if getattr(chunk, "channel", None) == "chat" and getattr(
                            chunk, "content", None
                        ):
                            active_recorder.first_visible_token()

                    result = yield from _yield_stream(
                        original(*args, **kwargs), observe
                    )
            except Exception:
                active_recorder.add_timing(timing_name, time.monotonic() - started)
                if request_stream:
                    active_recorder.finish_request(False)
                raise
            else:
                elapsed = time.monotonic() - started
                if request_stream:
                    active_recorder.finish_request(True)
                elif observe_generator:
                    active_recorder._generator_latency_ms = elapsed * 1000
                active_recorder.add_timing(timing_name, elapsed)
            return result

        wrapped._operations_wrapped = True  # type: ignore[attr-defined]
    else:

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            active_recorder = _ACTIVE_RECORDER.get() or recorder
            if timing_name == "retrieval":
                active_recorder.start_retrieval()
            started = time.monotonic()
            try:
                with active_recorder.activate(purpose):
                    result = original(*args, **kwargs)
                return result
            except Exception as exc:
                if external_provider:
                    model_name = getattr(target, "model_name", None)
                    active_recorder.record_model_call(
                        endpoint=external_provider,
                        model=str(model_name) if model_name else None,
                        call_id=len(active_recorder._calls) + 1,
                        started=started,
                        usage=None,
                        error=exc,
                        purpose=purpose,
                    )
                raise
            finally:
                active_recorder.add_timing(timing_name, time.monotonic() - started)
                if external_provider and "result" in locals():
                    model_name = getattr(target, "model_name", None)
                    active_recorder.record_model_call(
                        endpoint=external_provider,
                        model=str(model_name) if model_name else None,
                        call_id=len(active_recorder._calls) + 1,
                        started=started,
                        usage=None,
                        purpose=purpose,
                    )

        wrapped._operations_wrapped = True  # type: ignore[attr-defined]
    object.__setattr__(target, method_name, wrapped)


def _instrument_model_stream(
    target: Any, recorder: OperationsRecorder, timing_name: str
) -> None:
    original = getattr(target, "stream", None)
    if original is None or getattr(original, "_operations_wrapped", False):
        return

    def wrapped(*args: Any, **kwargs: Any) -> Iterator[Any]:
        active_recorder = _ACTIVE_RECORDER.get() or recorder
        started = time.monotonic()
        active_recorder._generator_started = started
        try:
            with active_recorder.activate(timing_name):
                result = yield from _yield_stream(original(*args, **kwargs))
            return result
        finally:
            active_recorder.add_timing(timing_name, time.monotonic() - started)

    wrapped._operations_wrapped = True  # type: ignore[attr-defined]
    object.__setattr__(target, "stream", wrapped)


def instrument_pipeline(pipeline: Any, recorder: OperationsRecorder) -> list[str]:
    """Attach timings at Kotaemon retrieval, evidence, and answer-stream boundaries."""
    from kotaemon.embeddings import BaseEmbeddings
    from kotaemon.indices.rankings import BaseReranking
    from kotaemon.indices.vectorindex import VectorRetrieval
    from ktem.index.file.pipelines import DocumentRetrievalPipeline

    model_names: list[str] = []

    def safe_model_metadata(model: Any) -> dict[str, Any]:
        fields = (
            "model",
            "model_name",
            "deployment_name",
            "base_url",
            "openai_api_base",
            "azure_endpoint",
            "timeout",
            "request_timeout",
            "max_retries",
            "max_tokens",
            "temperature",
        )
        metadata: dict[str, Any] = {"resource_type": type(model).__name__}
        for field in fields:
            value = getattr(model, field, None)
            if isinstance(value, (str, int, float, bool)):
                metadata[field] = value
        kwargs = getattr(model, "_kwargs", None)
        if isinstance(kwargs, dict):
            for field in fields:
                value = kwargs.get(field)
                if isinstance(value, (str, int, float, bool)):
                    metadata.setdefault(field, value)
        return metadata

    _instrument_method(
        pipeline,
        "stream",
        recorder,
        "pipeline",
        "application",
        stream=True,
        request_stream=True,
    )
    _instrument_method(pipeline, "retrieve", recorder, "retrieval", "retrieval")
    evidence = getattr(pipeline, "evidence_pipeline", None)
    if evidence is not None:
        _instrument_method(
            evidence,
            "run",
            recorder,
            "evidence_preparation",
            "evidence_preparation",
        )
    answering = getattr(pipeline, "answering_pipeline", None)
    if answering is not None:
        llm = getattr(answering, "llm", None)
        if llm is not None:
            recorder.model_metadata["generator"] = safe_model_metadata(llm)
            model_name = getattr(llm, "model", None) or getattr(llm, "model_name", None)
            if model_name:
                model_names.append(str(model_name))
            _instrument_model_stream(llm, recorder, "answer_generation")
        _instrument_method(
            answering,
            "stream",
            recorder,
            "answer_stream",
            "answer_generation",
            stream=True,
            observe_generator=False,
        )
        for citation_name in (
            "citation_pipeline",
            "create_mindmap_pipeline",
        ):
            citation = getattr(answering, citation_name, None)
            if citation is not None and hasattr(citation, "run"):
                _instrument_method(
                    citation,
                    "run",
                    recorder,
                    "citation_processing",
                    "citation_processing",
                )

    visited: set[int] = set()

    def instrument_retriever(value: Any) -> None:
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, (list, tuple)):
            for item in value:
                instrument_retriever(item)
            return
        if isinstance(value, DocumentRetrievalPipeline):
            vector_retrieval = getattr(value, "vector_retrieval", None)
            embedding = getattr(vector_retrieval, "embedding", None)
            recorder.model_metadata.setdefault("retrievers", []).append(
                {
                    "resource_type": type(value).__name__,
                    "top_k": getattr(value, "top_k", None),
                    "retrieval_mode": getattr(vector_retrieval, "retrieval_mode", None),
                    "embedding": (
                        safe_model_metadata(embedding)
                        if embedding is not None
                        else None
                    ),
                    "rerankers": [
                        safe_model_metadata(item)
                        for item in (getattr(value, "rerankers", []) or [])
                    ],
                }
            )
            _instrument_method(value, "run", recorder, "retriever", "retrieval")
            instrument_retriever(vector_retrieval)
            instrument_retriever(getattr(value, "rerankers", []))
            instrument_retriever(getattr(value, "llm_scorer", None))
        elif isinstance(value, VectorRetrieval):
            _instrument_method(value, "run", recorder, "vector_retrieval", "retrieval")
            instrument_retriever(getattr(value, "embedding", None))
            instrument_retriever(getattr(value, "vector_store", None))
            instrument_retriever(getattr(value, "doc_store", None))
            instrument_retriever(getattr(value, "rerankers", []))
        elif isinstance(value, BaseEmbeddings):
            method_name = (
                "run" if type(value).run is not BaseEmbeddings.run else "invoke"
            )
            if hasattr(value, method_name):
                _instrument_method(
                    value,
                    method_name,
                    recorder,
                    "query_embedding",
                    "query_embedding",
                )
        elif isinstance(value, BaseReranking):
            provider = (
                "cohere.rerank"
                if "cohere" in type(value).__module__.casefold()
                else None
            )
            _instrument_method(
                value,
                "run",
                recorder,
                "reranking",
                "reranking",
                external_provider=provider,
            )
            for name in ("llm",):
                instrument_retriever(getattr(value, name, None))
        elif value is not None and hasattr(value, "query"):
            _instrument_method(value, "query", recorder, "search", "search")

    instrument_retriever(getattr(pipeline, "retrievers", []))
    return model_names


@contextmanager
def instrument_openai_calls() -> Generator[None, None, None]:
    """Capture leaf OpenAI SDK calls, including supported streaming usage chunks."""
    import httpx

    from openai.resources.chat.completions import AsyncCompletions, Completions
    from openai.resources.embeddings import AsyncEmbeddings, Embeddings

    patched: list[tuple[type, str, Any]] = []
    call_id = 0
    original_httpx_send = httpx.Client.send
    original_async_httpx_send = httpx.AsyncClient.send

    def send(instance: Any, *args: Any, **kwargs: Any) -> Any:
        attempts = _ACTIVE_HTTP_ATTEMPTS.get()
        if attempts is not None:
            attempts[0] += 1
        return original_httpx_send(instance, *args, **kwargs)

    async def async_send(instance: Any, *args: Any, **kwargs: Any) -> Any:
        attempts = _ACTIVE_HTTP_ATTEMPTS.get()
        if attempts is not None:
            attempts[0] += 1
        return await original_async_httpx_send(instance, *args, **kwargs)

    httpx.Client.send = send
    httpx.AsyncClient.send = async_send

    def patch(cls: type, endpoint: str) -> None:
        nonlocal call_id
        original = cls.create

        def create(instance: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_id
            recorder = _ACTIVE_RECORDER.get()
            if recorder is None:
                return original(instance, *args, **kwargs)
            call_id += 1
            current_id = call_id
            started = time.monotonic()
            attempts = [0]
            attempts_token = _ACTIVE_HTTP_ATTEMPTS.set(attempts)
            model = kwargs.get("model")
            if endpoint == "openai.chat.completions" and kwargs.get("stream"):
                base_url = str(
                    getattr(getattr(instance, "_client", None), "base_url", "")
                )
                if "api.openai.com" in base_url:
                    stream_options = dict(kwargs.get("stream_options") or {})
                    stream_options["include_usage"] = True
                    kwargs["stream_options"] = stream_options
            try:
                response = original(instance, *args, **kwargs)
            except Exception as exc:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)
                recorder.record_model_call(
                    endpoint=endpoint,
                    model=str(model) if model else None,
                    call_id=current_id,
                    started=started,
                    usage=None,
                    error=exc,
                    http_attempts=attempts[0] or None,
                )
                raise
            else:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)

            if kwargs.get("stream"):

                def stream() -> Iterator[Any]:
                    final_usage = None
                    stream_error = None
                    try:
                        for item in response:
                            usage = _usage(item)
                            if usage is not None:
                                final_usage = usage
                            yield item
                    except Exception as exc:
                        stream_error = exc
                        raise
                    finally:
                        recorder.record_model_call(
                            endpoint=endpoint,
                            model=str(model) if model else None,
                            call_id=current_id,
                            started=started,
                            usage=final_usage,
                            error=stream_error,
                            http_attempts=attempts[0] or None,
                        )

                return stream()

            recorder.record_model_call(
                endpoint=endpoint,
                model=str(model) if model else None,
                call_id=current_id,
                started=started,
                usage=_usage(response),
                http_attempts=attempts[0] or None,
            )
            return response

        patched.append((cls, "create", original))
        setattr(cls, "create", create)

    def patch_parse(cls: type, endpoint: str) -> None:
        nonlocal call_id
        original = cls.parse

        def parse(instance: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_id
            recorder = _ACTIVE_RECORDER.get()
            if recorder is None:
                return original(instance, *args, **kwargs)
            call_id += 1
            current_id = call_id
            started = time.monotonic()
            attempts = [0]
            attempts_token = _ACTIVE_HTTP_ATTEMPTS.set(attempts)
            model = kwargs.get("model")
            try:
                response = original(instance, *args, **kwargs)
            except Exception as exc:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)
                recorder.record_model_call(
                    endpoint=endpoint,
                    model=str(model) if model else None,
                    call_id=current_id,
                    started=started,
                    usage=None,
                    error=exc,
                    http_attempts=attempts[0] or None,
                )
                raise
            else:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)
            recorder.record_model_call(
                endpoint=endpoint,
                model=str(model) if model else None,
                call_id=current_id,
                started=started,
                usage=_usage(response),
                http_attempts=attempts[0] or None,
            )
            return response

        patched.append((cls, "parse", original))
        setattr(cls, "parse", parse)

    def patch_async(cls: type, endpoint: str) -> None:
        nonlocal call_id
        original = cls.create

        async def create(instance: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_id
            recorder = _ACTIVE_RECORDER.get()
            if recorder is None:
                return await original(instance, *args, **kwargs)
            call_id += 1
            current_id = call_id
            started = time.monotonic()
            attempts = [0]
            attempts_token = _ACTIVE_HTTP_ATTEMPTS.set(attempts)
            model = kwargs.get("model")
            if endpoint == "openai.chat.completions" and kwargs.get("stream"):
                base_url = str(
                    getattr(getattr(instance, "_client", None), "base_url", "")
                )
                if "api.openai.com" in base_url:
                    stream_options = dict(kwargs.get("stream_options") or {})
                    stream_options["include_usage"] = True
                    kwargs["stream_options"] = stream_options
            try:
                response = await original(instance, *args, **kwargs)
            except Exception as exc:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)
                recorder.record_model_call(
                    endpoint=endpoint,
                    model=str(model) if model else None,
                    call_id=current_id,
                    started=started,
                    usage=None,
                    error=exc,
                    http_attempts=attempts[0] or None,
                    transport="async",
                )
                raise
            else:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)

            if kwargs.get("stream"):

                async def stream() -> AsyncIterator[Any]:
                    final_usage = None
                    stream_error = None
                    try:
                        async for item in response:
                            usage = _usage(item)
                            if usage is not None:
                                final_usage = usage
                            yield item
                    except Exception as exc:
                        stream_error = exc
                        raise
                    finally:
                        recorder.record_model_call(
                            endpoint=endpoint,
                            model=str(model) if model else None,
                            call_id=current_id,
                            started=started,
                            usage=final_usage,
                            error=stream_error,
                            http_attempts=attempts[0] or None,
                            transport="async",
                        )

                return stream()

            recorder.record_model_call(
                endpoint=endpoint,
                model=str(model) if model else None,
                call_id=current_id,
                started=started,
                usage=_usage(response),
                http_attempts=attempts[0] or None,
                transport="async",
            )
            return response

        patched.append((cls, "create", original))
        setattr(cls, "create", create)

    def patch_async_parse(cls: type, endpoint: str) -> None:
        nonlocal call_id
        original = cls.parse

        async def parse(instance: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal call_id
            recorder = _ACTIVE_RECORDER.get()
            if recorder is None:
                return await original(instance, *args, **kwargs)
            call_id += 1
            current_id = call_id
            started = time.monotonic()
            attempts = [0]
            attempts_token = _ACTIVE_HTTP_ATTEMPTS.set(attempts)
            model = kwargs.get("model")
            try:
                response = await original(instance, *args, **kwargs)
            except Exception as exc:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)
                recorder.record_model_call(
                    endpoint=endpoint,
                    model=str(model) if model else None,
                    call_id=current_id,
                    started=started,
                    usage=None,
                    error=exc,
                    http_attempts=attempts[0] or None,
                    transport="async",
                )
                raise
            else:
                _ACTIVE_HTTP_ATTEMPTS.reset(attempts_token)
            recorder.record_model_call(
                endpoint=endpoint,
                model=str(model) if model else None,
                call_id=current_id,
                started=started,
                usage=_usage(response),
                http_attempts=attempts[0] or None,
                transport="async",
            )
            return response

        patched.append((cls, "parse", original))
        setattr(cls, "parse", parse)

    patch(Completions, "openai.chat.completions")
    patch(Embeddings, "openai.embeddings")
    patch_parse(Completions, "openai.chat.completions")
    patch_async(AsyncCompletions, "openai.chat.completions")
    patch_async(AsyncEmbeddings, "openai.embeddings")
    patch_async_parse(AsyncCompletions, "openai.chat.completions")
    original_thread_start = threading.Thread.start

    def start_with_context(thread: threading.Thread, *args: Any, **kwargs: Any) -> Any:
        context = copy_context()
        original_run = thread.run

        def run_in_context(*run_args: Any, **run_kwargs: Any) -> Any:
            return context.run(original_run, *run_args, **run_kwargs)

        thread.run = run_in_context  # type: ignore[method-assign]
        return original_thread_start(thread, *args, **kwargs)

    threading.Thread.start = start_with_context
    try:
        yield
    finally:
        httpx.Client.send = original_httpx_send
        httpx.AsyncClient.send = original_async_httpx_send
        threading.Thread.start = original_thread_start
        for cls, method_name, original in reversed(patched):
            setattr(cls, method_name, original)
