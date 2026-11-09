#!/usr/bin/env python
"""Run retrieval claim evaluation for a golden dataset or selected questions."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

from evals.retriever.extract_claims import (
    DEFAULT_MODEL,
    _create_openai_client,
    _preview_record,
    _request_claims,
)
from evals.retriever.inspect_evidence import evidence_capture_session
from evals.retriever.judge_evidence import (
    _judge_question,
    _load_claims,
    _original_retrieval_limit,
    _score,
    _select_evidence,
)
from scripts.run_rag_eval import (
    _load_dataset_records,
    _validate_question_record,
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _artifact_stem(question_id: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("._")
    if not stem:
        raise ValueError(f"Question ID {question_id!r} cannot form an artifact name")
    return stem


def _failure_message(exc: Exception, stage: str) -> str:
    if stage in {"generate_claims", "judge"}:
        return f"OpenAI operation failed ({type(exc).__name__})."
    return f"{type(exc).__name__}: {exc}"


def _save_artifact(
    result: dict[str, Any],
    evidence_dump: dict[str, Any] | None,
    path: Path,
    stage: str | None = None,
    exc: Exception | None = None,
) -> None:
    if stage is not None and exc is not None:
        result["failure"] = {
            "stage": stage,
            "error": _failure_message(exc, stage),
        }
    _write_json(
        path,
        {
            "question_id": result["question_id"],
            "question": result.get("question"),
            "evidence_dump": evidence_dump,
            "evaluation": result,
        },
    )


def _ranked_evidence_with_metadata(
    question_id: str,
    dump: dict[str, Any],
    selected_text: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ranked_chunks = dump.get("ranked_chunks")
    if not isinstance(ranked_chunks, list):
        raise ValueError(f"{question_id} evidence dump has no ranked_chunks array")
    ordered_chunks = sorted(ranked_chunks, key=lambda chunk: chunk["rank"])
    chunks_by_rank: dict[int, list[dict[str, Any]]] = {}
    for chunk in ordered_chunks:
        chunks_by_rank.setdefault(chunk["rank"], []).append(chunk)

    enriched: list[dict[str, Any]] = []
    seen_by_rank: dict[int, int] = {}
    for selected in selected_text:
        rank = selected["rank"]
        matches = chunks_by_rank.get(rank, [])
        match_index = seen_by_rank.get(rank, 0)
        if match_index >= len(matches):
            raise ValueError(
                f"{question_id} selected rank {rank} does not match the saved "
                "ranked chunks"
            )
        chunk = matches[match_index]
        seen_by_rank[rank] = match_index + 1
        source = chunk.get("source")
        if not isinstance(source, dict):
            source = {}
        enriched.append(
            {
                "rank": rank,
                "text": selected["text"],
                "source_filename": source.get("filename"),
                "page": source.get("page"),
            }
        )
    return enriched


def _initial_result(
    question_id: str,
    question: str | None,
    evaluation_k: int | None,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "claims_source": None,
        "claims": [],
        "ranked_text_passed_to_judge": [],
        "verdicts": [],
        "useful_ranks": [],
        "available_chunk_count": None,
        "evaluated_chunk_count": None,
        "configured_retrieval_limit": None,
        "evaluation_mode": ("top_k" if evaluation_k is not None else "all_final"),
        "evaluation_k": evaluation_k,
        "precision_denominator": None,
        "claim_recall_at_k": None,
        "chunk_precision_at_k": None,
        "artifact_path": None,
    }


def _resolve_dataset_path(dataset: Path, repo_root: Path) -> Path:
    if dataset.is_absolute():
        return dataset
    if dataset.parent == Path("."):
        return repo_root / "data" / "golden" / dataset
    return Path.cwd() / dataset


def _load_selected_records(
    dataset_path: Path, question_ids: list[str] | None
) -> list[dict[str, Any]]:
    records = _load_dataset_records(dataset_path)
    if not records:
        raise ValueError(f"Dataset {dataset_path} contains no question records")

    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id, _ = _validate_question_record(record, dataset_path)
        if question_id in records_by_id:
            raise ValueError(f"Duplicate question ID {question_id!r} in {dataset_path}")
        records_by_id[question_id] = record

    if question_ids is None:
        selected_records = records
    else:
        missing_ids = [
            question_id
            for question_id in question_ids
            if question_id not in records_by_id
        ]
        if missing_ids:
            raise ValueError(
                f"Unknown question ID(s) in {dataset_path}: " + ", ".join(missing_ids)
            )
        selected_records = [records_by_id[question_id] for question_id in question_ids]

    for record in selected_records:
        if "retrieval_claims" in record:
            _load_claims(record, dataset_path)
    return selected_records


def _macro_average(rows: list[dict[str, Any]], metric_name: str) -> tuple[float, int]:
    values = [
        row[metric_name]
        for row in rows
        if row["status"] == "complete"
        and isinstance(row[metric_name], (int, float))
        and not isinstance(row[metric_name], bool)
    ]
    return (sum(values) / len(values) if values else 0.0, len(values))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrieval evaluation for every record in a golden dataset, "
            "or only the explicitly selected question IDs. Write one combined "
            "evidence and evaluation JSON artifact per question."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        default=None,
        metavar="ID",
        help=(
            "Question IDs to evaluate in the supplied order. Omit to evaluate "
            "every valid dataset entry in file order."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        metavar="K",
        help=(
            "Evaluate only the first K ranked chunks (positive integer). "
            "Omit to evaluate all chunks represented in the saved final context."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--user-id",
        help=(
            "Use this app user's saved Gradio settings (required if multiple "
            "users have saved settings)."
        ),
    )
    args = parser.parse_args(argv)

    if args.k is not None and args.k < 1:
        parser.error("--k must be a positive integer")
    if args.question_ids is not None and len(args.question_ids) != len(
        set(args.question_ids)
    ):
        parser.error("--question-ids must not contain duplicates")
    repo_root = Path(__file__).resolve().parents[2]
    dataset_path = _resolve_dataset_path(args.dataset, repo_root)
    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else Path.cwd() / args.output_dir
    )
    output_dir = output_dir.resolve()
    try:
        selected_records = _load_selected_records(dataset_path, args.question_ids)
        question_ids = [record["id"] for record in selected_records]
        stems = [_artifact_stem(question_id) for question_id in question_ids]
        if len(stems) != len(set(stems)):
            raise ValueError("Question IDs map to duplicate artifact filenames")
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    records_by_id = {record["id"]: record for record in selected_records}

    summary_rows: list[dict[str, Any]] = []
    work: list[dict[str, Any]] = []
    work_by_id: dict[str, dict[str, Any]] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, Path] = {}

    for question_id, stem in zip(question_ids, stems, strict=True):
        artifact_paths[question_id] = output_dir / f"{stem}_evaluation.json"
        result = _initial_result(question_id, None, args.k)
        result["artifact_path"] = str(artifact_paths[question_id])
        results_by_id[question_id] = result
        item = {
            "question_id": question_id,
            "result": result,
            "artifact_path": artifact_paths[question_id],
            "evidence_dump": None,
        }
        work.append(item)
        work_by_id[question_id] = item

    client = None
    for question_id in question_ids:
        record = records_by_id[question_id]
        result = results_by_id[question_id]
        result["question"] = record["question"]
        try:
            question = record["question"]
            if "retrieval_claims" in record:
                claims = _load_claims(record, dataset_path)
                result["claims_source"] = "stored"
            else:
                result["claims_source"] = "generated_draft"
                if client is None:
                    client = _create_openai_client()
                claim_texts = _request_claims(client, args.model, record)
                generated = _preview_record(question_id, question, claim_texts)
                claims = generated["retrieval_claims"]
            result["claims"] = claims
            _save_artifact(
                result,
                work_by_id[question_id]["evidence_dump"],
                artifact_paths[question_id],
            )
        except Exception as exc:
            if result["claims_source"] == "generated_draft":
                stage = "generate_claims"
            elif result["claims_source"] is None:
                stage = "load_question"
            else:
                stage = "load_claims"
            _save_artifact(
                result,
                work_by_id[question_id]["evidence_dump"],
                artifact_paths[question_id],
                stage,
                exc,
            )

    runnable = work
    if runnable:
        docs_dir = repo_root / "data" / "raw"
        try:
            with evidence_capture_session(docs_dir, args.user_id) as capture_question:
                for item in runnable:
                    question_id = item["question_id"]
                    result = item["result"]
                    record = records_by_id[question_id]
                    stage = "retrieve_evidence"
                    try:
                        dump = capture_question(question_id, record["question"])
                        item["evidence_dump"] = dump
                        stage = "select_evidence"
                        result["configured_retrieval_limit"] = (
                            _original_retrieval_limit(dump)
                        )
                        selected, available = _select_evidence(
                            question_id, dump, args.k
                        )
                        result["available_chunk_count"] = available
                        result["evaluated_chunk_count"] = len(selected)
                        result["precision_denominator"] = (
                            args.k if args.k is not None else len(selected) or None
                        )
                        selected_with_metadata = _ranked_evidence_with_metadata(
                            question_id, dump, selected
                        )
                        result["ranked_text_passed_to_judge"] = selected_with_metadata
                        _save_artifact(result, dump, item["artifact_path"])

                        if "failure" in result:
                            item["status"] = "failed"
                            item["failure"] = result["failure"]
                            continue

                        stage = "judge"
                        if result["claims"] and selected_with_metadata:
                            if client is None:
                                client = _create_openai_client()
                            judge_evidence = [
                                {"rank": chunk["rank"], "text": chunk["text"]}
                                for chunk in selected_with_metadata
                            ]
                            verdicts = _judge_question(
                                client,
                                args.model,
                                record["question"],
                                result["claims"],
                                judge_evidence,
                                question_id,
                            )
                        elif result["claims"]:
                            verdicts = [
                                {
                                    "claim_id": claim["id"],
                                    "verdict": "unsupported",
                                    "ranks": [],
                                }
                                for claim in result["claims"]
                            ]
                        else:
                            verdicts = []
                        metrics = _score(
                            result["claims"], verdicts, args.k, len(selected)
                        )
                        verdicts_by_id = {
                            verdict["claim_id"]: verdict for verdict in verdicts
                        }
                        result["verdicts"] = [
                            {
                                "claim_id": claim["id"],
                                "claim": claim["claim"],
                                "verdict": (
                                    verdicts_by_id[claim["id"]]["verdict"]
                                    if verdicts
                                    else None
                                ),
                                "ranks": (
                                    verdicts_by_id[claim["id"]]["ranks"]
                                    if verdicts
                                    else []
                                ),
                            }
                            for claim in result["claims"]
                        ]
                        if metrics is not None:
                            result["useful_ranks"] = metrics["useful_ranks"]
                            result["claim_recall_at_k"] = metrics["claim_recall_at_k"]
                            result["chunk_precision_at_k"] = metrics[
                                "chunk_precision_at_k"
                            ]
                        _save_artifact(result, dump, item["artifact_path"])
                        item["status"] = "complete"
                    except Exception as exc:
                        _save_artifact(
                            result,
                            item["evidence_dump"],
                            item["artifact_path"],
                            stage,
                            exc,
                        )
                        item["status"] = "failed"
                        item["failure"] = result["failure"]
        except Exception as exc:
            for item in runnable:
                if item.get("status") is not None:
                    continue
                result = item["result"]
                _save_artifact(
                    result,
                    item["evidence_dump"],
                    item["artifact_path"],
                    "setup",
                    exc,
                )
                item["status"] = "failed"
                item["failure"] = result["failure"]

    for item in work:
        question_id = item["question_id"]
        result = item["result"]
        status = item.get("status")
        if status is None:
            status = "failed" if "failure" in result else "complete"
        summary_rows.append(
            {
                "question_id": question_id,
                "status": status,
                "failure": result.get("failure"),
                "claim_recall_at_k": result["claim_recall_at_k"],
                "chunk_precision_at_k": result["chunk_precision_at_k"],
                "artifact_path": str(artifact_paths[question_id]),
            }
        )

    summary_path = output_dir / "summary.json"
    completed_count = sum(row["status"] == "complete" for row in summary_rows)
    failed_count = sum(row["status"] != "complete" for row in summary_rows)
    macro_recall, recall_question_count = _macro_average(
        summary_rows, "claim_recall_at_k"
    )
    macro_precision, precision_question_count = _macro_average(
        summary_rows, "chunk_precision_at_k"
    )
    try:
        _write_json(
            summary_path,
            {
                "dataset": str(dataset_path),
                "model": args.model,
                "total_questions": len(summary_rows),
                "completed_count": completed_count,
                "failed_count": failed_count,
                "evaluation_k": args.k,
                "evaluation_mode": ("top_k" if args.k is not None else "all_final"),
                "macro_average_claim_recall_at_k": macro_recall,
                "claim_recall_question_count": recall_question_count,
                "macro_average_chunk_precision_at_k": macro_precision,
                "chunk_precision_question_count": precision_question_count,
                "results": summary_rows,
            },
        )
    except OSError as exc:
        print(f"Error writing summary {summary_path}: {exc}", file=sys.stderr)
        return 1

    for row in summary_rows:
        print(
            f"{row['question_id']}: {row['status']}; "
            f"claim_recall_at_k={row['claim_recall_at_k']}; "
            f"chunk_precision_at_k={row['chunk_precision_at_k']}"
        )
    print(f"Output directory: {output_dir}")
    return 1 if any(row["status"] != "complete" for row in summary_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
