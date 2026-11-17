#!/usr/bin/env python
"""Run a single-question RAG triad evaluation using Kotaemon's live pipeline."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

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


def _sanitize_error(exc: Exception, api_key: str) -> str:
    message = str(exc)
    if api_key:
        message = message.replace(api_key, "[REDACTED]")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{type(exc).__name__}: {message}"


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
    api_key = ""
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
            answer_question = _prepare_question_runner(
                docs_dir,
                saved_settings,
                saved_index,
                evidence_callback=capture_final_context,
                stop_after_evidence=False,
            )

            for record, result, result_path in zip(
                records, results, result_paths, strict=True
            ):
                question_id = record["id"]
                capture_target.clear()

                stage = "kotaemon_run"
                try:
                    generated_answer, citations = answer_question(record["question"])
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
                        print(f"{metric.__class__.__name__}: {metric.score}")
                        print(f"Reason: {metric.reason}")

                    result["status"] = "complete"
                    _write_result(result_path, result)
                    print()
                except Exception as exc:
                    result["status"] = "failed"
                    result["failed_stage"] = stage
                    result["sanitized_error"] = _sanitize_error(exc, api_key)
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
            result["sanitized_error"] = _sanitize_error(exc, api_key)
            _write_result(result_path, result)

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
