#!/usr/bin/env python
"""Run a single-question RAG triad evaluation using Kotaemon's live pipeline."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import copy_context
from importlib.metadata import version
import json
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

from scripts.run_rag_eval import (
    _load_dataset_records,
    _prepare_question_runner,
    _read_saved_app_config,
    _seed_isolated_database,
    _set_isolated_runtime_paths,
    _validate_question_record,
)


JUDGE_MODEL = "gpt-4o-mini"


def _create_judge_model() -> Any:
    from decouple import config
    from deepeval.models import GPTModel

    api_key = config("OPENAI_API_KEY", default="").strip()
    if not api_key or api_key == "<YOUR_OPENAI_KEY>":
        raise ValueError("Set OPENAI_API_KEY in the environment or repo .env")

    return GPTModel(
        model=JUDGE_MODEL,  # gpt-4o-mini
        _openai_api_key=api_key,
    )


def _evaluate(
    question: str,
    generated_answer: str,
    final_context_texts: list[str],
    judge_model: Any,
) -> tuple[Any, list[tuple[str, Any]]]:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    test_case = LLMTestCase(
        input=question,
        actual_output=generated_answer,
        retrieval_context=final_context_texts,
    )
    metrics = [
        ("contextual_relevancy", ContextualRelevancyMetric(model=judge_model)),
        ("faithfulness", FaithfulnessMetric(model=judge_model)),
        ("answer_relevancy", AnswerRelevancyMetric(model=judge_model)),
    ]
    return test_case, metrics


def _write_result(path: Path | None, result: dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sanitize_error(exc: Exception, *api_keys: str) -> str:
    message = str(exc)
    for secret in api_keys:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{type(exc).__name__}: {message}"


def _update_feedback_status(result: dict[str, Any]) -> None:
    feedback = result["langsmith_feedback"]
    statuses = [item["status"] for item in feedback["metrics"].values()]
    if "failed" in statuses:
        feedback["status"] = "partial_failure"
    elif all(status == "uploaded" for status in statuses):
        feedback["status"] = "uploaded"
    elif any(status == "uploaded" for status in statuses):
        feedback["status"] = "incomplete"
    else:
        feedback["status"] = "pending"


def _upload_metric_feedback(
    client: Any,
    result: dict[str, Any],
    result_path: Path | None,
    metric_name: str,
    score: Any,
    reason: str | None,
    *api_keys: str,
) -> None:
    feedback_metric = result["langsmith_feedback"]["metrics"][metric_name]
    try:
        feedback = client.create_feedback(
            run_id=result["application_trace_id"],
            key=metric_name,
            score=round(float(score),4),
            comment=reason,
        )
    except Exception as exc:
        feedback_metric.update(
            status="failed",
            sanitized_error=_sanitize_error(exc, *api_keys),
        )
        _update_feedback_status(result)
        _write_result(result_path, result)
        print(
            f"[{result['question_id']}] Warning: LangSmith feedback upload "
            f"failed for {metric_name}: {feedback_metric['sanitized_error']}",
            file=sys.stderr,
        )
        return

    feedback_metric.update(
        status="uploaded",
        feedback_id=str(feedback.id),
        sanitized_error=None,
    )
    _update_feedback_status(result)
    _write_result(result_path, result)


def _trace_model_name(model: Any) -> str | None:
    if model is None:
        return None
    for attribute in ("model", "model_name", "deployment_name"):
        value = getattr(model, attribute, None)
        if value:
            return str(value)
    kwargs = getattr(model, "_kwargs", {})
    if isinstance(kwargs, dict):
        for attribute in ("model", "model_name", "deployment_name"):
            if kwargs.get(attribute):
                return str(kwargs[attribute])
    return type(model).__name__


def _trace_component_method(
    component: Any,
    method_name: str,
    *,
    name: str,
    client: Any,
    process_inputs: Callable[[dict[str, Any]], dict[str, Any]],
    process_outputs: Callable[[Any], Any] | None = None,
    reduce_fn: Callable[[Iterator[Any]], Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    from langsmith import traceable

    original = getattr(component, method_name)
    traced = traceable(
        name=name,
        run_type="chain",
        client=client,
        process_inputs=process_inputs,
        process_outputs=process_outputs,
        reduce_fn=reduce_fn,
        metadata=metadata,
    )(original)
    object.__setattr__(component, method_name, traced)


def _instrument_pipeline(pipeline: Any, client: Any) -> list[str]:
    retrieval = getattr(pipeline, "retrieve", None)
    evidence_pipeline = getattr(pipeline, "evidence_pipeline", None)
    answering_pipeline = getattr(pipeline, "answering_pipeline", None)
    if retrieval is None or evidence_pipeline is None or answering_pipeline is None:
        raise ValueError(
            "LangSmith tracing requires a pipeline with retrieval, evidence, "
            "and answering stages."
        )

    generator_name = _trace_model_name(getattr(answering_pipeline, "llm", None))
    model_names = [generator_name] if generator_name else []
    _trace_component_method(
        pipeline,
        "retrieve",
        name="retrieval",
        client=client,
        process_inputs=lambda inputs: {
            "query": inputs.get("message"),
            "history": inputs.get("history", []),
        },
        process_outputs=lambda output: {
            "retrieved_document_count": len(output[0]),
            "document_ids": [
                str(doc.doc_id) for doc in output[0] if getattr(doc, "doc_id", None)
            ],
        },
    )
    _trace_component_method(
        evidence_pipeline,
        "run",
        name="evidence preparation",
        client=client,
        process_inputs=lambda inputs: {
            "retrieved_document_count": len(inputs.get("docs", [])),
        },
        process_outputs=lambda output: {
            "evidence_mode": output.content[0],
            "final_context": output.content[1],
            "image_count": len(output.content[2]),
        },
    )

    def reduce_generation(chunks: Iterator[Any]) -> dict[str, str]:
        answer = "".join(
            str(chunk.content)
            for chunk in chunks
            if getattr(chunk, "channel", None) == "chat" and chunk.content is not None
        )
        return {"generated_answer": answer}

    _trace_component_method(
        answering_pipeline,
        "stream",
        name="generation",
        client=client,
        process_inputs=lambda inputs: {
            key: inputs[key]
            for key in ("question", "evidence", "evidence_mode")
            if key in inputs
        },
        reduce_fn=reduce_generation,
        metadata={"model_names": model_names},
    )
    return model_names


@contextmanager
def _instrument_generation_clients(client: Any) -> Generator[None, None, None]:
    from langsmith import traceable
    from langsmith.wrappers import wrap_openai
    from kotaemon.llms.chats.langchain_based import LCChatMixin
    from kotaemon.llms.chats.openai import AzureChatOpenAI, ChatOpenAI

    patched: list[tuple[type, str, Any]] = []

    def patch(cls: type, method_name: str, replacement: Callable[..., Any]) -> None:
        original = getattr(cls, method_name)
        patched.append((cls, method_name, original))
        setattr(cls, method_name, replacement(original))

    def wrap_prepare_client(original: Callable[..., Any]) -> Callable[..., Any]:
        def prepare_client(self: Any, *args: Any, **kwargs: Any) -> Any:
            return wrap_openai(original(self, *args, **kwargs))

        return prepare_client

    def wrap_openai_response(original: Callable[..., Any]) -> Callable[..., Any]:
        def openai_response(self: Any, client_: Any, **kwargs: Any) -> Any:
            if kwargs.get("stream") and self.base_url in (
                None,
                "https://api.openai.com/v1",
            ):
                kwargs.setdefault("stream_options", {"include_usage": True})
            return original(self, client_, **kwargs)

        return openai_response

    for cls in (ChatOpenAI, AzureChatOpenAI):
        patch(cls, "prepare_client", wrap_prepare_client)
    patch(ChatOpenAI, "openai_response", wrap_openai_response)

    def langchain_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        model = inputs.get("self")
        return {
            "model_name": _trace_model_name(model),
            "messages": inputs.get("messages"),
        }

    def langchain_outputs(output: Any) -> dict[str, Any]:
        if isinstance(output, dict):
            return output
        content = getattr(output, "content", None)
        if content is None:
            content = getattr(output, "text", "")
        result: dict[str, Any] = {"content": content}
        prompt_tokens = getattr(output, "prompt_tokens", -1)
        completion_tokens = getattr(output, "completion_tokens", -1)
        total_tokens = getattr(output, "total_tokens", -1)
        if prompt_tokens >= 0 and completion_tokens >= 0:
            result["usage_metadata"] = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": (
                    total_tokens
                    if total_tokens >= 0
                    else prompt_tokens + completion_tokens
                ),
            }
        return result

    def reduce_langchain_stream(chunks: Iterator[Any]) -> dict[str, Any]:
        return {
            "content": "".join(
                str(chunk.content)
                for chunk in chunks
                if getattr(chunk, "content", None) is not None
            )
        }

    def wrap_langchain_method(
        *, is_stream: bool = False
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def wrap(original: Callable[..., Any]) -> Callable[..., Any]:
            traced = traceable(
                name="LangChain generation",
                run_type="llm",
                client=client,
                process_inputs=langchain_inputs,
                process_outputs=langchain_outputs,
                reduce_fn=reduce_langchain_stream if is_stream else None,
            )(original)

            def invoke(self: Any, *args: Any, **kwargs: Any) -> Any:
                extra = {
                    "metadata": {"model_name": _trace_model_name(self)},
                }
                return traced(self, *args, langsmith_extra=extra, **kwargs)

            if is_stream:

                def stream(self: Any, *args: Any, **kwargs: Any) -> Any:
                    extra = {
                        "metadata": {"model_name": _trace_model_name(self)},
                    }
                    return traced(self, *args, langsmith_extra=extra, **kwargs)

                return stream
            return invoke

        return wrap

    for method_name in ("invoke", "ainvoke"):
        patch(LCChatMixin, method_name, wrap_langchain_method())
    for method_name in ("stream", "astream"):
        patch(LCChatMixin, method_name, wrap_langchain_method(is_stream=True))

    try:
        yield
    finally:
        for cls, method_name, original in reversed(patched):
            setattr(cls, method_name, original)


@contextmanager
def _propagate_trace_context_to_threads() -> Generator[None, None, None]:
    original_start = threading.Thread.start

    def start_with_context(thread: threading.Thread, *args: Any, **kwargs: Any) -> None:
        context = copy_context()
        original_run = thread.run

        def run_in_context(*run_args: Any, **run_kwargs: Any) -> Any:
            return context.run(original_run, *run_args, **run_kwargs)

        setattr(thread, "run", run_in_context)
        original_start(thread, *args, **kwargs)

    setattr(threading.Thread, "start", start_with_context)
    try:
        yield
    finally:
        setattr(threading.Thread, "start", original_start)


def _run_traced_question(
    question: str,
    question_id: str,
    dataset: Path,
    answer_question: Callable[[str], tuple[str, str]],
    capture_target: dict[str, Any],
    client: Any,
    project_name: str,
    result: dict[str, Any],
    result_path: Path | None,
) -> tuple[str, str]:
    from langsmith import traceable, tracing_context
    from langsmith.run_helpers import get_current_run_tree

    @traceable(
        name="Kotaemon request",
        run_type="chain",
        client=client,
        project_name=project_name,
        process_inputs=lambda inputs: {
            "query": inputs["question"],
            "question_id": inputs["question_id"],
            "dataset": inputs["dataset_path"],
        },
    )
    def run_request(
        question: str, question_id: str, dataset_path: str
    ) -> dict[str, Any]:
        root_run = get_current_run_tree()
        if root_run is None:
            raise RuntimeError("LangSmith did not create the application root trace.")
        capture_target["trace_root"] = root_run
        generated_answer, citations = answer_question(question)
        if "final_context" not in capture_target:
            raise RuntimeError(
                "Kotaemon completed without capturing its final generator context."
            )
        return {
            "query": question,
            "question_id": question_id,
            "dataset": dataset_path,
            "final_context": capture_target["final_context"],
            "generated_answer": generated_answer,
            "model_names": capture_target.get("model_names", []),
            "citations": citations,
        }

    try:
        with tracing_context(
            enabled=True,
            client=client,
            project_name=project_name,
            metadata={"question_id": question_id, "dataset": str(dataset)},
        ):
            output = run_request(question, question_id, str(dataset))
        return output["generated_answer"], output["citations"]
    finally:
        root_run = capture_target.get("trace_root")
        if root_run is not None:
            result["application_trace_id"] = str(root_run.id)
            result["application_trace_url"] = client.get_run_url(
                run=root_run,
                project_name=project_name,
            )
            _write_result(result_path, result)


def _resolve_dataset_path(dataset: Path, repo_root: Path) -> Path:
    if dataset.is_absolute():
        return dataset
    if dataset.parent == Path("."):
        return repo_root / "data" / "golden" / dataset
    return Path.cwd() / dataset


def _load_selected_records(
    dataset: Path,
    question_id: str | None,
    question_ids: list[str] | None,
) -> list[dict[str, Any]]:
    records = _load_dataset_records(dataset)
    if not records:
        raise ValueError(f"Dataset {dataset} contains no question records")

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        current_id, _ = _validate_question_record(record, dataset)
        if current_id in records_by_id:
            raise ValueError(f"Duplicate question ID {current_id!r} in {dataset}")
        records_by_id[current_id] = record

    selected_ids = [question_id] if question_id is not None else question_ids
    if selected_ids is None:
        return records
    missing_ids = [item for item in selected_ids if item not in records_by_id]
    if missing_ids:
        raise ValueError(
            f"Unknown question ID(s) in {dataset}: " + ", ".join(missing_ids)
        )
    return [records_by_id[item] for item in selected_ids]


def _artifact_stem(question_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("._")
    if not stem:
        raise ValueError(f"Question ID {question_id!r} cannot form an artifact name")
    return stem


def _initial_result(
    dataset: Path, record: dict[str, Any], result_path: Path | None
) -> dict[str, Any]:
    return {
        "dataset_path": str(dataset),
        "question_id": record["id"],
        "query_text": record["question"],
        "generated_answer": None,
        "retrieval_context": [],
        "retrieved_document_count": 0,
        "context_block_count": 0,
        "image_count": 0,
        "text_only_evaluation": True,
        "judge_model": JUDGE_MODEL,
        "deepeval_version": version("deepeval"),
        "status": "pending_kotaemon",
        "metrics": {
            name: {"score": None, "reason": None}
            for name in (
                "contextual_relevancy",
                "faithfulness",
                "answer_relevancy",
            )
        },
        "result_path": str(result_path) if result_path is not None else None,
    }


def _build_summary(
    dataset: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metric_names = (
        "contextual_relevancy",
        "faithfulness",
        "answer_relevancy",
    )
    means = {}
    for name in metric_names:
        scores = [
            row["metrics"][name]["score"]
            for row in rows
            if row["status"] == "complete"
            and isinstance(row["metrics"][name]["score"], (int, float))
            and not isinstance(row["metrics"][name]["score"], bool)
        ]
        means[name] = {
            "mean": sum(scores) / len(scores) if scores else None,
            "question_count": len(scores),
        }
    return {
        "dataset_path": str(dataset),
        "total_questions": len(rows),
        "completed_count": sum(row["status"] == "complete" for row in rows),
        "failed_count": sum(row["status"] != "complete" for row in rows),
        "metrics": means,
        "results": [
            {
                "question_id": row["question_id"],
                "status": row["status"],
                "result_path": row["result_path"],
                "failed_stage": row.get("failed_stage"),
                "sanitized_error": row.get("sanitized_error"),
            }
            for row in rows
        ],
    }


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--question-id")
    selection.add_argument("--question-ids", nargs="+", metavar="ID")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for per-question JSON results and the batch summary.",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the exact retrieval_context passed to DeepEval before scoring.",
    )
    parser.add_argument(
        "--langsmith",
        action="store_true",
        help="Trace each Kotaemon request to the configured LangSmith project.",
    )
    parser.add_argument(
        "--user-id",
        help=(
            "Use this app user's saved Gradio settings (required if multiple "
            "users have saved settings)."
        ),
    )
    args = parser.parse_args(argv)

    dataset = _resolve_dataset_path(args.dataset, repo_root)
    output_dir = (
        args.output_dir
        if args.output_dir is None or args.output_dir.is_absolute()
        else Path.cwd() / args.output_dir
    )
    batch_mode = args.question_id is None
    docs_dir = repo_root / "data" / "raw"
    pdfs = sorted(docs_dir.glob("*.pdf")) if docs_dir.is_dir() else []
    try:
        records = _load_selected_records(
            dataset,
            args.question_id,
            args.question_ids,
        )
        if len(args.question_ids or []) != len(set(args.question_ids or [])):
            raise ValueError("--question-ids must not contain duplicates")
        stems = [_artifact_stem(record["id"]) for record in records]
        if len(stems) != len(set(stems)):
            raise ValueError("Question IDs map to duplicate artifact filenames")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not pdfs:
        print(f"Error: no PDF files found in {docs_dir}", file=sys.stderr)
        return 1
    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(
                f"Error creating output directory {output_dir}: {exc}", file=sys.stderr
            )
            return 1

    result_paths = [
        output_dir / f"{stem}_result.json" if output_dir is not None else None
        for stem in stems
    ]
    results = [
        _initial_result(dataset, record, result_path)
        for record, result_path in zip(records, result_paths, strict=True)
    ]
    if args.langsmith:
        for result in results:
            result.update(application_trace_id=None, application_trace_url=None)
            result["langsmith_feedback"] = {
                "status": "pending",
                "metrics": {
                    name: {
                        "status": "pending",
                        "feedback_id": None,
                        "sanitized_error": None,
                    }
                    for name in (
                        "contextual_relevancy",
                        "faithfulness",
                        "answer_relevancy",
                    )
                },
            }
    api_key = ""
    langsmith_api_key = ""
    langsmith_project = ""
    langsmith_client = None
    capture_target: dict[str, Any] = {}

    def capture_final_context(
        docs: list[Any],
        prepared_content: Any,
        retrievers: list[Any],
        context_limit: int,
    ) -> None:
        try:
            evidence_mode, final_context, images = prepared_content
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Kotaemon's prepared evidence did not contain context and images"
            ) from exc
        if not isinstance(final_context, str):
            raise TypeError("Kotaemon's prepared evidence context is not text")
        _ = (retrievers, context_limit, evidence_mode)
        capture_target["final_context"] = final_context
        capture_target["retrieved_document_count"] = len(docs)
        capture_target["images"] = images

    try:
        from decouple import config

        api_key = config("OPENAI_API_KEY", default="").strip()
        if args.langsmith:
            from langsmith import Client

            langsmith_api_key = config("LANGSMITH_API_KEY", default="").strip()
            if not langsmith_api_key:
                raise ValueError(
                    "Set LANGSMITH_API_KEY in the environment or repo .env "
                    "to use --langsmith."
                )
            langsmith_endpoint = config("LANGSMITH_ENDPOINT", default="").strip()
            langsmith_project = config(
                "LANGSMITH_PROJECT", default="kotaemon-pipeline-eval"
            ).strip()
            if not langsmith_project:
                raise ValueError("LANGSMITH_PROJECT must not be empty.")
            langsmith_client = Client(
                api_url=langsmith_endpoint or None,
                api_key=langsmith_api_key,
            )
        from theflow.settings import settings

        saved_tables, saved_settings, saved_index = _read_saved_app_config(
            settings.KH_DATABASE,
            args.user_id,
            getattr(settings, "KH_FEATURE_USER_MANAGEMENT", False),
        )
        with tempfile.TemporaryDirectory(prefix="kotaemon-pipeline-eval-") as temp_dir:
            runtime_data = Path(temp_dir)
            _set_isolated_runtime_paths(settings, runtime_data)
            db_path = Path(settings.KH_DATABASE.removeprefix("sqlite:///"))
            _seed_isolated_database(db_path, saved_tables, saved_settings)
            pipeline_callback = None
            if langsmith_client is not None:

                def instrument_pipeline(pipeline: Any) -> None:
                    capture_target["model_names"] = _instrument_pipeline(
                        pipeline, langsmith_client
                    )

                pipeline_callback = instrument_pipeline

            answer_question = _prepare_question_runner(
                docs_dir,
                saved_settings,
                saved_index,
                evidence_callback=capture_final_context,
                stop_after_evidence=False,
                pipeline_callback=pipeline_callback,
            )

            for record, result, result_path in zip(
                records, results, result_paths, strict=True
            ):
                question_id = record["id"]
                capture_target.clear()

                stage = "kotaemon_run"
                try:
                    if langsmith_client is None:
                        generated_answer, citations = answer_question(
                            record["question"]
                        )
                    else:
                        with (
                            _instrument_generation_clients(langsmith_client),
                            _propagate_trace_context_to_threads(),
                        ):
                            generated_answer, citations = _run_traced_question(
                                record["question"],
                                question_id,
                                dataset,
                                answer_question,
                                capture_target,
                                langsmith_client,
                                langsmith_project,
                                result,
                                result_path,
                            )
                        print(f"LangSmith trace: {result['application_trace_id']}")
                        print(f"LangSmith URL: {result['application_trace_url']}")
                    _ = citations
                    if "final_context" not in capture_target:
                        raise RuntimeError(
                            "Kotaemon completed without capturing its final "
                            "generator context."
                        )
                    final_context = capture_target["final_context"]
                    final_context_texts = (
                        [final_context] if final_context.strip() else []
                    )
                    image_count = len(capture_target.get("images") or [])
                    result.update(
                        generated_answer=generated_answer,
                        retrieval_context=final_context_texts,
                        retrieved_document_count=capture_target[
                            "retrieved_document_count"
                        ],
                        context_block_count=len(final_context_texts),
                        image_count=image_count,
                        status="pending_scoring",
                    )
                    _write_result(result_path, result)
                    if args.show_context:
                        print(f"{question_id} DeepEval retrieval_context:")
                        print(
                            json.dumps(
                                final_context_texts,
                                ensure_ascii=False,
                                indent=2,
                            )
                        )
                        sys.stdout.flush()

                    print(f"Question ID: {question_id}")
                    print("Generated answer:")
                    print(generated_answer)
                    print(f"Context blocks evaluated: {len(final_context_texts)}")
                    print(
                        "Retrieved documents captured: "
                        f"{result['retrieved_document_count']}"
                    )
                    print(
                        "Text-only evaluation: yes; "
                        "Kotaemon also passed images to the generator: "
                        f"{'yes' if image_count else 'no'}"
                        + (f" ({image_count})" if image_count else "")
                    )

                    stage = "judge_setup"
                    judge_model = _create_judge_model()
                    test_case, metrics = _evaluate(
                        record["question"],
                        generated_answer,
                        final_context_texts,
                        judge_model,
                    )
                    for name, metric in metrics:
                        stage = f"scoring:{name}"
                        metric.measure(test_case)
                        score = metric.score
                        if hasattr(score, "item"):
                            score = score.item()
                        result["metrics"][name] = {
                            "score": score,
                            "reason": metric.reason,
                        }
                        _write_result(result_path, result)
                        if langsmith_client is not None:
                            _upload_metric_feedback(
                                langsmith_client,
                                result,
                                result_path,
                                name,
                                score,
                                metric.reason,
                                api_key,
                                langsmith_api_key,
                            )
                        print(f"{metric.__class__.__name__}: {metric.score}")
                        print(f"Reason: {metric.reason}")

                    result["status"] = "complete"
                    _write_result(result_path, result)
                    print()
                except Exception as exc:
                    result["status"] = "failed"
                    result["failed_stage"] = stage
                    result["sanitized_error"] = _sanitize_error(
                        exc, api_key, langsmith_api_key
                    )
                    _write_result(result_path, result)
                    print(
                        f"[{question_id}] Error ({stage}): "
                        f"{result['sanitized_error']}",
                        file=sys.stderr,
                    )
    except Exception as exc:
        print(f"Error initializing Kotaemon: {exc}", file=sys.stderr)
        for record, result, result_path in zip(
            records, results, result_paths, strict=True
        ):
            if result["status"] in ("failed", "complete"):
                continue
            result["status"] = "failed"
            result["failed_stage"] = "kotaemon_setup"
            result["sanitized_error"] = _sanitize_error(exc, api_key, langsmith_api_key)
            _write_result(result_path, result)

    if langsmith_client is not None:
        try:
            tracing_queue = langsmith_client.tracing_queue
            if tracing_queue is not None:
                tracing_queue.join()
            langsmith_client.cleanup()
        except Exception as exc:
            print(f"Error flushing LangSmith traces: {exc}", file=sys.stderr)
            return 1

    if output_dir is not None and batch_mode:
        summary_path = output_dir / "summary.json"
        summary = _build_summary(dataset, results)
        try:
            _write_result(summary_path, summary)
        except OSError as exc:
            print(f"Error writing summary {summary_path}: {exc}", file=sys.stderr)
            return 1
        print(
            f"Batch summary: {summary['completed_count']} succeeded, "
            f"{summary['failed_count']} failed."
        )
        print(f"Summary JSON: {summary_path}")

    if output_dir is not None and not batch_mode:
        print(f"JSON result: {result_paths[0]}")
    return 1 if any(result["status"] != "complete" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
