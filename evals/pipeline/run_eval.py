#!/usr/bin/env python
"""Run a single-question RAG triad evaluation using Kotaemon's live pipeline."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from scripts.run_rag_eval import (
    _load_question,
    _prepare_question_runner,
    _read_saved_app_config,
    _seed_isolated_database,
    _set_isolated_runtime_paths,
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


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--question-id", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for the single-question JSON result.",
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

    dataset = args.dataset if args.dataset.is_absolute() else Path.cwd() / args.dataset
    output_dir = (
        args.output_dir
        if args.output_dir is None or args.output_dir.is_absolute()
        else Path.cwd() / args.output_dir
    )
    result_path: Path | None = None
    if output_dir is not None:
        result_path = output_dir / f"{args.question_id}_result.json"
    docs_dir = repo_root / "data" / "raw"
    pdfs = sorted(docs_dir.glob("*.pdf")) if docs_dir.is_dir() else []
    if not pdfs:
        print(f"Error: no PDF files found in {docs_dir}", file=sys.stderr)
        return 1

    try:
        record = _load_question(dataset, args.question_id)
        question = record["question"]
        captured: dict[str, Any] = {}

        def capture_final_context(
            docs: list[Any],
            prepared_content: Any,
            _retrievers: list[Any],
            _context_limit: int,
        ) -> None:
            try:
                _evidence_mode, final_context, images = prepared_content
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "Kotaemon's prepared evidence did not contain context and images"
                ) from exc
            if not isinstance(final_context, str):
                raise TypeError("Kotaemon's prepared evidence context is not text")
            captured["final_context"] = final_context
            captured["retrieved_document_count"] = len(docs)
            captured["images"] = images

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
            generated_answer, _citations = answer_question(question)

        if "final_context" not in captured:
            raise RuntimeError(
                "Kotaemon completed without capturing its final generator context."
            )
        final_context = captured["final_context"]
        final_context_texts = [final_context] if final_context.strip() else []
        image_count = len(captured.get("images") or [])
        if result_path is not None:
            result_path.parent.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {
            "dataset_path": str(dataset),
            "question_id": args.question_id,
            "query_text": question,
            "generated_answer": generated_answer,
            "retrieval_context": final_context_texts,
            "retrieved_document_count": captured["retrieved_document_count"],
            "context_block_count": len(final_context_texts),
            "image_count": image_count,
            "text_only_evaluation": True,
            "judge_model": JUDGE_MODEL,
            "deepeval_version": version("deepeval"),
            "status": "pending_scoring",
            "metrics": {
                name: {"score": None, "reason": None}
                for name in (
                    "contextual_relevancy",
                    "faithfulness",
                    "answer_relevancy",
                )
            },
        }
        _write_result(result_path, result)
        if args.show_context:
            print("DeepEval retrieval_context:")
            print(json.dumps(final_context_texts, ensure_ascii=False, indent=2))
            sys.stdout.flush()

        api_key = ""
        current_stage = "judge_setup"
        try:
            from decouple import config

            api_key = config("OPENAI_API_KEY", default="").strip()
            judge_model = _create_judge_model()
            current_stage = "scoring"
            test_case, metrics = _evaluate(
                question,
                generated_answer,
                final_context_texts,
                judge_model,
            )
            for name, metric in metrics:
                current_stage = f"scoring:{name}"
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
        except Exception as exc:
            result["status"] = "failed"
            result["failed_stage"] = current_stage
            result["sanitized_error"] = _sanitize_error(exc, api_key)
            _write_result(result_path, result)
            print(f"Error: {result['sanitized_error']}", file=sys.stderr)
            return 1

        result["status"] = "complete"
        _write_result(result_path, result)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Question ID: {args.question_id}")
    print("Generated answer:")
    print(generated_answer)
    print(f"Context blocks evaluated: {len(final_context_texts)}")
    print("Retrieved documents captured: " f"{captured['retrieved_document_count']}")
    print(
        "Text-only evaluation: yes; "
        f"Kotaemon also passed images to the generator: {'yes' if image_count else 'no'}"
        + (f" ({image_count})" if image_count else "")
    )
    if result_path is not None:
        print(f"JSON result: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
