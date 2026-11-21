#!/usr/bin/env python
"""Evaluate one saved pipeline answer against its golden question."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from decouple import config
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from evals.application.run_eval import (
    JUDGE_MODEL,
    _create_openai_client,
    _find_question,
    _read_json,
    _request_completeness,
    _sanitize_error,
    _validate_pipeline_result,
    _write_json,
)
from evals.operations import (
    OperationsRecorder,
    build_batch_operations_summary,
    instrument_openai_calls,
)


CLAIM_EXTRACTION_PROMPT = """\
Extract a concise list of distinct factual claims explicitly stated or implied
by the generated answer. Use the question to resolve references and establish
scope. Do not use the golden answer or outside knowledge. Write each claim as a
self-contained statement. Preserve entities, values, units, attribution,
relationships, and uncertainty. Split independently verifiable facts, but keep
qualifiers needed to state each fact accurately. Omit opinions, filler, and
duplicate claims. Return only factual claims and return an empty list if there
are none.
"""

REFERENCE_SUPPORT_PROMPT = """\
Judge each factual claim using only the supplied golden_answer as the reference.
Do not use outside knowledge. Mark a claim supported only when the reference
establishes it, allowing equivalent wording and derivations. Mark contradicted
only when the reference explicitly conflicts with the claim, including a
conflicting value, unit, attribution, or qualifier. A claim that is absent from
the reference or not established by it is unverified, not contradicted. Judge
each claim independently and give a concise reason. Return exactly one verdict
for every supplied claim ID.
"""

ANSWER_RELEVANCE_PROMPT = """\
Judge whether each supplied claim helps answer the question. Use required_points
when present to clarify expected scope, while accepting equivalent relevant
answers. Judge relevance only: do not assess whether the claim is true or
supported. A factually wrong claim may still be relevant if it addresses what
the question asks. Give a concise reason and return exactly one judgment for
every supplied claim ID.
"""


class ClaimExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[str] = Field(description="Distinct factual claims from the answer.")


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    verdict: Literal["supported", "contradicted", "unverified"]
    reason: str


class SupportJudgments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[ClaimVerdict]


class RelevanceJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str
    relevant: bool
    reason: str


class RelevanceJudgments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgments: list[RelevanceJudgment]


def _artifact_path(output_dir: Path, question_id: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("._")
    if not stem:
        raise ValueError(f"Question ID {question_id!r} cannot form an artifact name")
    return output_dir / f"{stem}_generator_result.json"


def _validated_claims(claims: Any, question_id: str) -> list[str]:
    if not isinstance(claims, list):
        raise ValueError(f"Extracted claims for {question_id} must be a JSON array")

    validated: list[str] = []
    seen: set[str] = set()
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(
                f"Extracted claim {index} for {question_id} must be non-empty text"
            )
        claim = " ".join(claim.split())
        normalized = claim.casefold().strip(" .")
        if not normalized:
            raise ValueError(f"Extracted claim {index} for {question_id} is empty")
        if normalized in seen:
            raise ValueError(f"Duplicate extracted claim for {question_id}: {claim!r}")
        seen.add(normalized)
        validated.append(claim)
    return validated


def _request_claims(
    client: OpenAI,
    model: str,
    question: str,
    generated_answer: str,
) -> list[str]:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": CLAIM_EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "generated_answer": generated_answer},
                    ensure_ascii=False,
                ),
            },
        ],
        response_format=ClaimExtraction,
        temperature=0,
    )
    if not response.choices:
        raise ValueError("OpenAI returned no claim-extraction choices")
    message = response.choices[0].message
    if message.refusal:
        raise ValueError("OpenAI refused claim extraction")
    if message.parsed is None:
        raise ValueError("OpenAI returned no structured claim extraction")
    return _validated_claims(message.parsed.claims, "selected question")


def _validate_judgments(
    judgments: list[Any], claim_ids: list[str], stage: str
) -> list[Any]:
    found: dict[str, Any] = {}
    for judgment in judgments:
        claim_id = judgment.claim_id.strip()
        if claim_id in found:
            raise ValueError(f"{stage} returned duplicate judgment for {claim_id}")
        found[claim_id] = judgment
    if set(found) != set(claim_ids):
        missing = [claim_id for claim_id in claim_ids if claim_id not in found]
        extra = [claim_id for claim_id in found if claim_id not in claim_ids]
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"unexpected: {', '.join(extra)}")
        raise ValueError(f"{stage} claim IDs do not match ({'; '.join(details)})")
    return [found[claim_id] for claim_id in claim_ids]


def _request_support(
    client: OpenAI,
    model: str,
    golden_answer: str,
    claims: list[dict[str, str]],
) -> list[ClaimVerdict]:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": REFERENCE_SUPPORT_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"golden_answer": golden_answer, "claims": claims},
                    ensure_ascii=False,
                ),
            },
        ],
        response_format=SupportJudgments,
        temperature=0,
    )
    if not response.choices:
        raise ValueError("OpenAI returned no reference-support choices")
    message = response.choices[0].message
    if message.refusal:
        raise ValueError("OpenAI refused reference-support judgment")
    if message.parsed is None:
        raise ValueError("OpenAI returned no structured reference-support judgments")
    return _validate_judgments(
        message.parsed.judgments,
        [claim["claim_id"] for claim in claims],
        "Reference-support judgment",
    )


def _request_relevance(
    client: OpenAI,
    model: str,
    question: str,
    required_points: Any,
    claims: list[dict[str, str]],
) -> list[RelevanceJudgment]:
    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": ANSWER_RELEVANCE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": question,
                        "required_points": required_points,
                        "claims": claims,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        response_format=RelevanceJudgments,
        temperature=0,
    )
    if not response.choices:
        raise ValueError("OpenAI returned no answer-relevance choices")
    message = response.choices[0].message
    if message.refusal:
        raise ValueError("OpenAI refused answer-relevance judgment")
    if message.parsed is None:
        raise ValueError("OpenAI returned no structured answer-relevance judgments")
    return _validate_judgments(
        message.parsed.judgments,
        [claim["claim_id"] for claim in claims],
        "Answer-relevance judgment",
    )


def _evaluate(
    repo_root: Path,
    question_id: str,
    pipeline_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_path = _artifact_path(output_dir, question_id)
    result: dict[str, Any] = {
        "question_id": question_id,
        "source_pipeline_result": str(pipeline_path),
        "trace_id": None,
        "matched_dataset_filename": None,
        "question": None,
        "generated_answer": None,
        "golden_answer": None,
        "required_points": None,
        "extracted_claims": [],
        "reference_support": None,
        "answer_relevance": None,
        "completeness": None,
        "judge_model": JUDGE_MODEL,
        "completed_stages": [],
        "status": "pending",
        "result_path": str(output_path),
    }
    operations = OperationsRecorder()
    result["operations"] = operations.snapshot()

    api_key = ""
    stage = "load_question"
    try:
        record, dataset_path = _find_question(repo_root, question_id)
        result.update(
            matched_dataset_filename=dataset_path.name,
            question=record["question"],
            golden_answer=record["golden_answer"],
            required_points=record.get("required_points"),
        )
        result["completed_stages"].append("load_question")

        stage = "load_pipeline_result"
        answer, query, pipeline_data = _validate_pipeline_result(
            question_id, record, pipeline_path
        )
        result.update(
            generated_answer=answer,
            query=query,
            trace_id=(
                str(
                    pipeline_data.get("application_trace_id")
                    or pipeline_data.get("trace_id")
                )
                if pipeline_data.get("application_trace_id")
                or pipeline_data.get("trace_id")
                else None
            ),
        )
        result["operations"] = operations.snapshot()
        result["operations"]["originating_run"] = pipeline_data.get("operations")
        result["completed_stages"].append("load_pipeline_result")
        _write_json(output_path, result)

        stage = "judge_setup"
        api_key = config("OPENAI_API_KEY", default="").strip()
        client = _create_openai_client()
        operations.start_evaluation()
        result["completed_stages"].append("judge_setup")

        stage = "extract_claims"
        with instrument_openai_calls(), operations.activate("evaluation"):
            claims = _request_claims(client, JUDGE_MODEL, record["question"], answer)
        claim_items = [
            {"claim_id": f"{question_id}-G{index:02d}", "claim": claim}
            for index, claim in enumerate(claims, start=1)
        ]
        result["extracted_claims"] = claim_items
        result["operations"] = operations.snapshot()
        result["operations"]["originating_run"] = pipeline_data.get("operations")
        result["completed_stages"].append(stage)
        _write_json(output_path, result)

        stage = "reference_support"
        with instrument_openai_calls(), operations.activate("evaluation"):
            support = (
                _request_support(
                    client, JUDGE_MODEL, record["golden_answer"], claim_items
                )
                if claim_items
                else []
            )
        support_by_id = {item.claim_id: item for item in support}
        for claim in result["extracted_claims"]:
            support_item = support_by_id[claim["claim_id"]]
            claim.update(
                reference_verdict=support_item.verdict,
                reference_reason=support_item.reason.strip(),
            )
        support_count = sum(item.verdict == "supported" for item in support)
        factual_count = len(claim_items)
        result["reference_support"] = {
            "numerator": support_count,
            "denominator": factual_count,
            "score": support_count / factual_count if factual_count else None,
            "reason": (
                None
                if factual_count
                else "No factual claims were extracted, so the rate is undefined."
            ),
        }
        result["completed_stages"].append(stage)
        result["operations"] = operations.snapshot()
        result["operations"]["originating_run"] = pipeline_data.get("operations")
        _write_json(output_path, result)

        stage = "answer_relevance"
        with instrument_openai_calls(), operations.activate("evaluation"):
            relevance = (
                _request_relevance(
                    client,
                    JUDGE_MODEL,
                    record["question"],
                    record.get("required_points"),
                    claim_items,
                )
                if claim_items
                else []
            )
        relevance_by_id = {item.claim_id: item for item in relevance}
        for claim in result["extracted_claims"]:
            relevance_item = relevance_by_id[claim["claim_id"]]
            claim.update(
                relevant=relevance_item.relevant,
                relevance_reason=relevance_item.reason.strip(),
            )
        relevant_count = sum(item.relevant for item in relevance)
        result["answer_relevance"] = {
            "numerator": relevant_count,
            "denominator": len(claim_items),
            "score": relevant_count / len(claim_items) if claim_items else None,
            "reason": (
                None
                if claim_items
                else "No claims were extracted, so the rate is undefined."
            ),
        }
        result["completed_stages"].append(stage)
        result["operations"] = operations.snapshot()
        result["operations"]["originating_run"] = pipeline_data.get("operations")
        _write_json(output_path, result)

        stage = "completeness"
        with instrument_openai_calls(), operations.activate("evaluation"):
            requirements = _request_completeness(client, record, answer)
        met_count = sum(item["verdict"] == "met" for item in requirements)
        partial_requirements = [
            item for item in requirements if item["verdict"] == "partial"
        ]
        result["completeness"] = {
            "requirements": requirements,
            "numerator": met_count,
            "denominator": len(requirements),
            "score": met_count / len(requirements),
            "met_count": met_count,
            "partial_count": len(partial_requirements),
            "unmet_count": sum(item["verdict"] == "unmet" for item in requirements),
            "partial_requirements": partial_requirements,
        }
        result["completed_stages"].append(stage)
        operations.finish_evaluation()
        result["operations"] = operations.snapshot()
        result["operations"]["originating_run"] = pipeline_data.get("operations")
        result["status"] = "complete"
        operations.measure_serialization(result)
        _write_json(output_path, result)
    except Exception as exc:
        operations.finish_evaluation()
        result["status"] = "failed"
        result["failed_stage"] = stage
        result["sanitized_error"] = _sanitize_error(exc, api_key)
        result["operations"] = operations.snapshot()
        if "pipeline_data" in locals():
            result["operations"]["originating_run"] = pipeline_data.get("operations")
        operations.measure_serialization(result)
        try:
            _write_json(output_path, result)
        except OSError as save_exc:
            print(
                f"Could not save failure result {output_path}: "
                f"{_sanitize_error(save_exc, api_key)}",
                file=sys.stderr,
            )
        print(
            f"[{question_id}] Error ({stage}): {result['sanitized_error']}",
            file=sys.stderr,
        )
    return result


def _discover_pipeline_results(
    results_dir: Path,
) -> tuple[dict[str, list[Path]], list[str]]:
    if not results_dir.is_dir():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    results: dict[str, list[Path]] = {}
    errors: list[str] = []
    for path in sorted(results_dir.glob("*.json")):
        if path.stem.casefold() == "summary":
            continue
        try:
            data = _read_json(path, "pipeline result")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if not isinstance(data, dict):
            continue
        question_id = data.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            continue
        results.setdefault(question_id, []).append(path)
    return results, errors


def _write_discovery_failure(
    output_dir: Path,
    question_id: str,
    message: str,
) -> dict[str, Any]:
    path = _artifact_path(output_dir, question_id)
    result = {
        "question_id": question_id,
        "source_pipeline_result": None,
        "trace_id": None,
        "matched_dataset_filename": None,
        "question": None,
        "generated_answer": None,
        "golden_answer": None,
        "required_points": None,
        "extracted_claims": [],
        "reference_support": None,
        "answer_relevance": None,
        "completeness": None,
        "judge_model": JUDGE_MODEL,
        "completed_stages": [],
        "status": "failed",
        "failed_stage": "discovery",
        "sanitized_error": message,
        "result_path": str(path),
        "operations": {
            **OperationsRecorder().snapshot(),
            "requests": {"status": "not_started", "failed_stage": "discovery"},
        },
    }
    _write_json(path, result)
    print(f"[{question_id}] Error (discovery): {message}", file=sys.stderr)
    return result


def _build_batch_summary(
    results: list[dict[str, Any]],
    discovery_errors: list[str],
) -> dict[str, Any]:
    completed = [item for item in results if item["status"] == "complete"]
    metrics: dict[str, dict[str, float | int | None]] = {}
    for metric in ("reference_support", "answer_relevance", "completeness"):
        scores = [
            item[metric]["score"]
            for item in completed
            if isinstance(item.get(metric), dict)
            and isinstance(item[metric].get("score"), (int, float))
            and not isinstance(item[metric].get("score"), bool)
        ]
        metrics[metric] = {
            "mean": sum(scores) / len(scores) if scores else None,
            "question_count": len(scores),
        }
    return {
        "total_questions": len(results),
        "completed_count": len(completed),
        "failed_count": sum(item["status"] != "complete" for item in results)
        + len(discovery_errors),
        "metrics": metrics,
        "operations": build_batch_operations_summary(results),
        "results": [
            {
                "question_id": item["question_id"],
                "status": item["status"],
                "result_path": item.get("result_path"),
                "reference_support_score": (item.get("reference_support") or {}).get(
                    "score"
                ),
                "answer_relevance_score": (item.get("answer_relevance") or {}).get(
                    "score"
                ),
                "completeness_score": (item.get("completeness") or {}).get("score"),
                "failed_stage": item.get("failed_stage"),
                "sanitized_error": item.get("sanitized_error"),
            }
            for item in results
        ],
        "discovery_errors": discovery_errors,
    }


def _run_batch(
    repo_root: Path,
    results_dir: Path,
    question_ids: list[str] | None,
    output_dir: Path,
) -> int:
    try:
        found, discovery_errors = _discover_pipeline_results(results_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    evaluated: list[dict[str, Any]] = []
    duplicate_requests: set[str] = set()
    if question_ids is not None and len(question_ids) != len(set(question_ids)):
        duplicate_requests = {
            question_id
            for question_id in question_ids
            if question_ids.count(question_id) > 1
        }
        for question_id in sorted(duplicate_requests):
            evaluated.append(
                _write_discovery_failure(
                    output_dir,
                    question_id,
                    f"Question ID {question_id!r} was requested more than once.",
                )
            )
        requested = list(dict.fromkeys(question_ids))
    else:
        requested = question_ids

    selected_ids = (
        [item for item in requested if item not in duplicate_requests]
        if requested is not None
        else sorted(found)
    )
    for question_id in selected_ids:
        paths = found.get(question_id, [])
        if not paths:
            message = (
                f"No pipeline result found for requested question ID "
                f"{question_id!r} in {results_dir}"
            )
            evaluated.append(_write_discovery_failure(output_dir, question_id, message))
            continue
        if len(paths) > 1:
            message = (
                f"Duplicate pipeline results for question ID {question_id!r}: "
                + ", ".join(str(path) for path in paths)
            )
            evaluated.append(_write_discovery_failure(output_dir, question_id, message))
            continue
        evaluated.append(_evaluate(repo_root, question_id, paths[0], output_dir))

    if not evaluated and not discovery_errors:
        discovery_errors.append(f"No pipeline result JSON files found in {results_dir}")
        print(f"Error: {discovery_errors[-1]}", file=sys.stderr)
    for error in discovery_errors:
        print(f"Error: {error}", file=sys.stderr)

    summary = _build_batch_summary(evaluated, discovery_errors)
    summary_path = output_dir / "summary.json"
    try:
        _write_json(summary_path, summary)
    except OSError as exc:
        print(f"Error writing batch summary {summary_path}: {exc}", file=sys.stderr)
        return 1
    print(
        f"Batch summary: {summary['completed_count']} succeeded, "
        f"{summary['failed_count']} failed."
    )
    print(f"JSON summary: {summary_path}")
    return 1 if summary["failed_count"] else 0


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question-id")
    parser.add_argument("--result", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--question-ids", nargs="+", metavar="ID")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    single_mode = args.question_id is not None or args.result is not None
    batch_mode = args.results_dir is not None or args.question_ids is not None
    if single_mode and batch_mode:
        parser.error(
            "Use either --question-id/--result or --results-dir/--question-ids"
        )
    if single_mode and (args.question_id is None or args.result is None):
        parser.error("--question-id and --result must be supplied together")
    if not single_mode and args.results_dir is None:
        parser.error("Batch mode requires --results-dir")

    output_dir = (
        args.output_dir
        if args.output_dir.is_absolute()
        else Path.cwd() / args.output_dir
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error creating output directory {output_dir}: {exc}", file=sys.stderr)
        return 1

    if batch_mode:
        results_dir = (
            args.results_dir
            if args.results_dir.is_absolute()
            else Path.cwd() / args.results_dir
        )
        return _run_batch(repo_root, results_dir, args.question_ids, output_dir)

    question_id = args.question_id.strip()
    if not question_id:
        parser.error("--question-id must not be empty")
    pipeline_path = (
        args.result if args.result.is_absolute() else Path.cwd() / args.result
    )
    result = _evaluate(repo_root, question_id, pipeline_path, output_dir)
    print(f"JSON result: {result['result_path']}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
