#!/usr/bin/env python
"""Evaluate saved Kotaemon pipeline answers for correctness and completeness."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal

from decouple import config
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


JUDGE_MODEL = "gpt-4o-mini"
GOLDEN_FILENAMES = ("aop.json", "rlpm.json", "shiro.json", "target.json")

CORRECTNESS_SYSTEM_PROMPT = """\
Judge a generated answer against the question, golden answer, and answerability
metadata. Use the golden answer and metadata as the reference for factual truth.
Accept equivalent wording, valid derivations, correct units, and well-supported
uncertainty. Check factual agreement, values, units, attribution, and
uncertainty. Omissions are completeness, not correctness, except that an answer
containing only refusal or filler is incorrect when the reference provides an
answer.

Do not turn extra, non-contradictory detail into a factual error unless it
asserts a fact that conflicts with or is unsupported by the supplied reference
and metadata. Extraneous detail belongs to completeness only when it is an
essential requirement; otherwise ignore it for the correctness judgment.

Return exactly:
- verdict: correct, partially_correct, or incorrect
- reason: a concise factual explanation
- factual_errors: specific incorrect or unsupported factual claims

Use partially_correct when the answer has substantial correct information but
also has a material factual error or omits a core answer. Use incorrect for
fundamentally wrong, contradictory, or refusal/filler answers."""

COMPLETENESS_SYSTEM_PROMPT = """\
Assess a generated answer against an exact list of essential requirements.
When required_points is supplied, transform it into a distinct, independently
assessable requirement list. Preserve uncertainty requirements and avoid
duplicates. When required_points is absent, derive essential requirements from
the question and golden answer. Make every requirement atomic, but keep needed
qualifiers (entity, value, unit, uncertainty, attribution) in the same
requirement. Do not add optional examples or duplicative requirements.
Do not convert grading instructions such as "do not invent", "do not confuse",
or "label as inference" into requirements. Do not add a requirement that the
answer must include no extra detail unless it is an essential answer constraint.
When required_points is absent, derive only positive factual requirements
directly needed to answer the question; never invent a requirement about
excluding irrelevant but factual detail.

For every requirement return an exact requirement string and:
- met: the requirement is fully answered
- partial: part but not all is answered
- unmet: the requirement is absent or contradicted
- reason: a concise explanation

Do not assign partial to simple wording differences or equivalent derivations.
Use the exact requirement text in the returned list."""


class CorrectnessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["correct", "partially_correct", "incorrect"]
    reason: str
    factual_errors: list[str]


class CompletenessRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement_id: str
    requirement: str
    verdict: Literal["met", "partial", "unmet"]
    reason: str


class CompletenessOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[CompletenessRequirement]


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON ({path}): {exc}") from exc


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


def _sanitize_error(exc: Exception, *api_keys: str) -> str:
    message = str(exc)
    for secret in api_keys:
        if secret:
            message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(r"\bBearer\s+\S+", "Bearer [REDACTED]", message)
    message = re.sub(r"\s+", " ", message).strip()
    return f"{type(exc).__name__}: {message}"


def _create_openai_client() -> OpenAI:
    api_key = config("OPENAI_API_KEY", default="").strip()
    if not api_key or api_key == "<YOUR_OPENAI_KEY>":
        raise ValueError("OPENAI_API_KEY is not configured in the environment or .env")
    base_url = (
        config("OPENAI_API_BASE", default="").strip()
        or "https://api.openai.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _find_question(
    repo_root: Path, question_id: str
) -> tuple[dict[str, Any], Path]:
    matches: list[tuple[dict[str, Any], Path]] = []
    golden_dir = repo_root / "data" / "golden"
    for filename in GOLDEN_FILENAMES:
        path = golden_dir / filename
        if not path.is_file():
            continue
        data = _read_json(path, "golden dataset")
        if not isinstance(data, list):
            raise ValueError(f"Golden dataset {path} must contain a JSON array")
        for record in data:
            if isinstance(record, dict) and record.get("id") == question_id:
                matches.append((record, path))

    if not matches:
        raise ValueError(
            f"Question ID {question_id!r} is missing from "
            "aop.json, rlpm.json, shiro.json, and target.json"
        )
    if len(matches) > 1:
        locations = ", ".join(str(path) for _, path in matches)
        raise ValueError(
            f"Question ID {question_id!r} is duplicated across golden datasets: "
            f"{locations}"
        )
    record, path = matches[0]
    question = record.get("question")
    golden_answer = record.get("golden_answer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Question {question_id} has no non-empty question")
    if not isinstance(golden_answer, str) or not golden_answer.strip():
        raise ValueError(f"Question {question_id} has no non-empty golden_answer")
    return record, path


def _validate_pipeline_result(
    question_id: str,
    record: dict[str, Any],
    pipeline_path: Path,
) -> tuple[str, str, dict[str, Any]]:
    data = _read_json(pipeline_path, "pipeline result")
    if not isinstance(data, dict):
        raise ValueError(f"Pipeline result {pipeline_path} must be a JSON object")

    result_question_id = data.get("question_id")
    if result_question_id != question_id:
        raise ValueError(
            f"Pipeline result question ID mismatch: expected {question_id!r}, "
            f"found {result_question_id!r}"
        )

    query = data.get("query_text")
    if query is None:
        query = data.get("query")
    if query is None:
        query = data.get("question")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            f"Pipeline result {pipeline_path} has no valid query_text/query/question"
        )
    if query.strip() != record["question"].strip():
        raise ValueError(
            "Pipeline result query does not match the selected golden question: "
            f"expected {record['question']!r}, found {query!r}"
        )

    answer = data.get("generated_answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(
            f"Pipeline result {pipeline_path} has no non-empty generated_answer"
        )
    return answer.strip(), query.strip(), data


def _references(record: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "golden_answer",
        "evidence",
        "answerability",
        "required_points",
        "question_type",
        "difficulty",
    )
    return {key: record[key] for key in keys if key in record}


def _request_correctness(
    client: OpenAI,
    record: dict[str, Any],
    generated_answer: str,
    retrieval_context: Any,
) -> CorrectnessOutput:
    user_input = {
        "question": record["question"],
        "golden_answer": record["golden_answer"],
        "generated_answer": generated_answer,
        "answerability": record.get("answerability"),
        "question_type": record.get("question_type"),
        "evidence": record.get("evidence"),
        "required_points": record.get("required_points"),
        "retrieval_context": retrieval_context,
    }
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": CORRECTNESS_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
        ],
        response_format=CorrectnessOutput,
        temperature=0,
    )
    if not response.choices:
        raise ValueError("OpenAI returned no correctness choices")
    message = response.choices[0].message
    if message.refusal:
        raise ValueError("OpenAI refused the correctness judgment")
    if message.parsed is None:
        raise ValueError("OpenAI returned no structured correctness judgment")
    return message.parsed


def _normalize_requirement(requirement: str) -> str:
    return " ".join(requirement.casefold().split()).strip(" .")


def _validate_requirements(
    question_id: str, output: CompletenessOutput
) -> list[dict[str, Any]]:
    if not output.requirements:
        raise ValueError(f"Completeness judge returned no requirements for {question_id}")

    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(output.requirements, start=1):
        requirement = item.requirement.strip()
        normalized = _normalize_requirement(requirement)
        requirement_id = item.requirement_id.strip()
        negative_requirement = bool(
            re.match(
                r"^(the answer (must|should) not|answer (must|should) not|"
                r"do not|does not|must not|"
                r"should not|avoid|not )",
                normalized,
            )
        )
        if negative_requirement:
            continue
        if not requirement or not normalized:
            raise ValueError(
                f"Completeness requirement {index} for {question_id} is empty"
            )
        if normalized in seen:
            raise ValueError(
                f"Duplicate completeness requirement for {question_id}: {requirement!r}"
            )
        if not requirement_id:
            requirement_id = f"{question_id}-R{index:02d}"
        if requirement_id in used_ids:
            raise ValueError(
                f"Duplicate completeness requirement ID for {question_id}: "
                f"{requirement_id}"
            )
        seen.add(normalized)
        used_ids.add(requirement_id)
        validated.append(
            {
                "requirement_id": requirement_id,
                "requirement": requirement,
                "verdict": item.verdict,
                "reason": item.reason.strip(),
            }
        )
    if not validated:
        raise ValueError(
            f"Completeness judge returned no positive requirements for {question_id}"
        )
    return validated


def _request_completeness(
    client: OpenAI,
    record: dict[str, Any],
    generated_answer: str,
) -> list[dict[str, Any]]:
    user_input = {
        "question": record["question"],
        "golden_answer": record["golden_answer"],
        "generated_answer": generated_answer,
        "answerability": record.get("answerability"),
        "question_type": record.get("question_type"),
        "required_points": record.get("required_points"),
    }
    response = client.beta.chat.completions.parse(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": COMPLETENESS_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_input, ensure_ascii=False)},
        ],
        response_format=CompletenessOutput,
        temperature=0,
    )
    if not response.choices:
        raise ValueError("OpenAI returned no completeness choices")
    message = response.choices[0].message
    if message.refusal:
        raise ValueError("OpenAI refused the completeness judgment")
    if message.parsed is None:
        raise ValueError("OpenAI returned no structured completeness judgment")
    return _validate_requirements(record["id"], message.parsed)


def _result_path(output_dir: Path, question_id: str) -> Path:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", question_id).strip("._")
    if not stem:
        raise ValueError(f"Question ID {question_id!r} cannot form an artifact name")
    return output_dir / f"{stem}_application_result.json"


def _evaluate_saved_result(
    repo_root: Path,
    question_id: str,
    pipeline_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_path = _result_path(output_dir, question_id)
    result: dict[str, Any] = {
        "question_id": question_id,
        "source_pipeline_result": str(pipeline_path),
        "matched_dataset_filename": None,
        "matched_dataset_path": None,
        "question": None,
        "query": None,
        "generated_answer": None,
        "references": {},
        "judge_model": JUDGE_MODEL,
        "trace_id": None,
        "trace_url": None,
        "pipeline_status": None,
        "correctness": None,
        "correctness_score": None,
        "completeness": None,
        "completeness_score": None,
        "status": "pending",
    }

    api_key = ""
    stage = "load_question"
    try:
        record, matched_dataset = _find_question(repo_root, question_id)
        stage = "load_pipeline_result"
        answer, query, pipeline_data = _validate_pipeline_result(
            question_id, record, pipeline_path
        )
        result.update(
            matched_dataset_filename=matched_dataset.name,
            matched_dataset_path=str(matched_dataset),
            question=record["question"],
            query=query,
            generated_answer=answer,
            references=_references(record),
            trace_id=(
                str(pipeline_data["application_trace_id"])
                if pipeline_data.get("application_trace_id")
                else None
            ),
            trace_url=(
                str(pipeline_data["application_trace_url"])
                if pipeline_data.get("application_trace_url")
                else None
            ),
            pipeline_status=pipeline_data.get("status"),
        )
        _write_json(output_path, result)

        stage = "judge_setup"
        api_key = config("OPENAI_API_KEY", default="").strip()
        client = _create_openai_client()

        stage = "correctness"
        correctness = _request_correctness(
            client,
            record,
            answer,
            pipeline_data.get("retrieval_context"),
        )
        if (
            correctness.verdict == "partially_correct"
            and not correctness.factual_errors
        ):
            correctness.verdict = "correct"
        correctness_score = {
            "correct": 1.0,
            "partially_correct": 0.5,
            "incorrect": 0.0,
        }[correctness.verdict]
        result["correctness"] = {
            "verdict": correctness.verdict,
            "reason": correctness.reason.strip(),
            "factual_errors": [
                error.strip() for error in correctness.factual_errors if error.strip()
            ],
        }
        result["correctness_score"] = correctness_score
        _write_json(output_path, result)

        stage = "completeness"
        requirements = _request_completeness(client, record, answer)
        met_count = sum(item["verdict"] == "met" for item in requirements)
        partial = [item for item in requirements if item["verdict"] == "partial"]
        completeness_score = met_count / len(requirements)
        result["completeness"] = {
            "requirements": requirements,
            "met_count": met_count,
            "partial_count": len(partial),
            "unmet_count": sum(
                item["verdict"] == "unmet" for item in requirements
            ),
            "total_requirements": len(requirements),
            "partial_requirements": partial,
        }
        result["completeness_score"] = completeness_score
        result["status"] = "complete"
        _write_json(output_path, result)
        print(
            f"[{question_id}] correctness={correctness.verdict} "
            f"({correctness_score:g}), completeness={completeness_score:g} "
            f"({met_count}/{len(requirements)} met)"
        )
    except Exception as exc:
        result["status"] = "failed"
        result["failed_stage"] = stage
        result["sanitized_error"] = _sanitize_error(exc, api_key)
        try:
            _write_json(output_path, result)
        except OSError as save_exc:
            print(
                f"[{question_id}] Could not save failure result {output_path}: "
                f"{_sanitize_error(save_exc, api_key)}",
                file=sys.stderr,
            )
        print(
            f"[{question_id}] Error ({stage}): {result['sanitized_error']}",
            file=sys.stderr,
        )
    result["result_path"] = str(output_path)
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
        if not isinstance(data.get("query_text"), str):
            continue
        results.setdefault(question_id, []).append(path)
    return results, errors


def _write_discovery_failure(
    output_dir: Path,
    question_id: str,
    message: str,
) -> dict[str, Any]:
    path = _result_path(output_dir, question_id)
    result = {
        "question_id": question_id,
        "source_pipeline_result": None,
        "matched_dataset_filename": None,
        "matched_dataset_path": None,
        "question": None,
        "query": None,
        "generated_answer": None,
        "references": {},
        "judge_model": JUDGE_MODEL,
        "trace_id": None,
        "trace_url": None,
        "pipeline_status": None,
        "correctness": None,
        "correctness_score": None,
        "completeness": None,
        "completeness_score": None,
        "status": "failed",
        "failed_stage": "discovery",
        "sanitized_error": message,
        "result_path": str(path),
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
    for metric in ("correctness", "completeness"):
        score_key = f"{metric}_score"
        scores = [
            item[score_key]
            for item in completed
            if isinstance(item.get(score_key), (int, float))
            and not isinstance(item.get(score_key), bool)
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
        "results": [
            {
                "question_id": item["question_id"],
                "status": item["status"],
                "result_path": item.get("result_path"),
                "correctness_score": item.get("correctness_score"),
                "completeness_score": item.get("completeness_score"),
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

    output_dir.mkdir(parents=True, exist_ok=True)
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
            evaluated.append(
                _write_discovery_failure(output_dir, question_id, message)
            )
            continue
        if len(paths) > 1:
            message = (
                f"Duplicate pipeline results for question ID {question_id!r}: "
                + ", ".join(str(path) for path in paths)
            )
            evaluated.append(
                _write_discovery_failure(output_dir, question_id, message)
            )
            continue
        evaluated.append(
            _evaluate_saved_result(repo_root, question_id, paths[0], output_dir)
        )

    if not evaluated and not discovery_errors:
        discovery_errors.append(
            f"No pipeline result JSON files found in {results_dir}"
        )
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
        parser.error("Use either --question-id/--result or --results-dir/--question-ids")
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
    pipeline_path = (
        args.result if args.result.is_absolute() else Path.cwd() / args.result
    )
    result = _evaluate_saved_result(
        repo_root, question_id, pipeline_path, output_dir
    )
    print(f"JSON result: {result['result_path']}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
