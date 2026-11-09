#!/usr/bin/env python
"""Extract, review, and apply retrieval claims from golden answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from decouple import config
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field

from scripts.run_rag_eval import (
    _load_dataset_records,
    _validate_question_record,
)


DEFAULT_MODEL = "gpt-4o-mini"
PREVIEW_VERSION = 1

SYSTEM_PROMPT = """\
Extract the essential factual claims that a retriever should surface evidence
for. Use the question to establish scope. Use required_points, when present,
to select relevant facts, and use golden_answer to supply their exact content.
When required_points is absent, select the essential facts from golden_answer.

Write each claim as a self-contained factual statement that a passage or a
small set of passages could verify. Preserve the entities, relationships,
values, units, and scope. Split facts that can be verified independently.
Keep qualifiers together when they are needed to state one fact accurately.
Do not add facts or duplicate them.

Skip grading instructions such as "does not invent values" or "labels the
answer as an inference." Skip optional examples and facts inferred by the
answer author. Skip claims that require checking the entire corpus for an
omission, such as "no exact value is reported" or "no benchmark value is
given." If the answer also states how the value is defined or determined,
extract that factual statement when it is relevant to the question.

Before returning, check each candidate: could a retrieved passage explicitly
state this fact or provide enough information to verify it? Remove candidates
that fail this check. Return an empty list if no claims qualify. Return only
the required structured output.
"""


class ClaimExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claims: list[str] = Field(
        description=(
            "Distinct, self-contained, independently verifiable positive "
            "facts "
            "relevant to answering the question."
        )
    )


def _create_openai_client() -> OpenAI:
    api_key = config("OPENAI_API_KEY", default="")
    if not api_key or api_key == "<YOUR_OPENAI_KEY>":
        raise ValueError(
            "OPENAI_API_KEY is not configured in the environment or .env"
        )
    base_url = (
        config("OPENAI_API_BASE", default="")
        or "https://api.openai.com/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


def _normalize_claim(claim: str) -> str:
    return " ".join(claim.casefold().split()).strip(" .")


def _validate_claims(claims: Any, question_id: str) -> list[str]:
    if not isinstance(claims, list):
        raise ValueError(f"Claims for {question_id} must be a JSON array")

    validated: list[str] = []
    seen: set[str] = set()
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError(
                f"Claim {index} for {question_id} must be a non-empty string"
            )
        claim = claim.strip()
        normalized = _normalize_claim(claim)
        if not normalized:
            raise ValueError(f"Claim {index} for {question_id} is empty")
        if normalized in seen:
            raise ValueError(f"Duplicate claim for {question_id}: {claim!r}")
        seen.add(normalized)
        validated.append(claim)
    return validated


def _load_selected_records(
    dataset: Path, question_ids: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not question_ids:
        raise ValueError("Provide at least one question ID")
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Question IDs must not contain duplicates")

    records = _load_dataset_records(dataset)
    found: dict[str, dict[str, Any]] = {}
    for record in records:
        question_id = record.get("id")
        if question_id in question_ids:
            if question_id in found:
                raise ValueError(
                    f"Question ID {question_id!r} occurs more than once in "
                    f"{dataset}"
                )
            _validate_question_record(record, dataset)
            found[question_id] = record

    missing = [
        question_id for question_id in question_ids if question_id not in found
    ]
    if missing:
        raise ValueError(
            f"Question ID(s) not found in {dataset}: {', '.join(missing)}"
        )
    return records, [found[question_id] for question_id in question_ids]


def _request_claims(
    client: OpenAI, model: str, record: dict[str, Any]
) -> list[str]:
    question_id, question = _validate_question_record(
        record, Path("<dataset>")
    )
    golden_answer = record.get("golden_answer")
    if not isinstance(golden_answer, str) or not golden_answer.strip():
        raise ValueError(
            f"Question {question_id} has no non-empty golden_answer"
        )

    user_input = {
        "question": question,
        "golden_answer": golden_answer,
    }
    required_points = record.get("required_points")
    if required_points is not None:
        if not isinstance(required_points, str):
            raise ValueError(
                f"required_points for {question_id} must be a string "
                "when present"
            )
        user_input["required_points"] = required_points

    response = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(user_input, ensure_ascii=False),
            },
        ],
        response_format=ClaimExtraction,
        temperature=0,
    )
    if not response.choices:
        raise ValueError(
            f"OpenAI returned no choices for question {question_id}"
        )
    message = response.choices[0].message
    if message.refusal:
        raise ValueError(f"OpenAI refused claim extraction for {question_id}")
    if message.parsed is None:
        raise ValueError(
            f"OpenAI returned no structured claims for question {question_id}"
        )
    return _validate_claims(message.parsed.claims, question_id)


def _preview_record(
    question_id: str, question: str, claims: list[str]
) -> dict:
    return {
        "question_id": question_id,
        "question": question,
        "retrieval_claims": [
            {"id": f"{question_id}-C{index:02d}", "claim": claim}
            for index, claim in enumerate(claims, start=1)
        ],
    }


def _make_preview(
    model: str,
    selected_records: list[dict[str, Any]],
    claim_lists: list[list[str]],
) -> dict:
    return {
        "version": PREVIEW_VERSION,
        "model": model,
        "records": [
            _preview_record(record["id"], record["question"], claims)
            for record, claims in zip(
                selected_records, claim_lists, strict=True
            )
        ],
    }


def _validate_preview(
    preview_path: Path, question_ids: list[str]
) -> list[dict[str, Any]]:
    try:
        preview = json.loads(preview_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"Cannot read approved preview {preview_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Approved preview is not valid JSON: {preview_path}: {exc}"
        ) from exc

    if (
        not isinstance(preview, dict)
        or preview.get("version") != PREVIEW_VERSION
    ):
        raise ValueError(
            "Approved preview has an unsupported or missing version"
        )
    if not isinstance(preview.get("model"), str) or not preview["model"]:
        raise ValueError("Approved preview must include a non-empty model")
    records = preview.get("records")
    if not isinstance(records, list):
        raise ValueError("Approved preview must include a records array")

    approved: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every approved preview record must be an object")
        question_id = record.get("question_id")
        question = record.get("question")
        if not isinstance(question_id, str) or not question_id:
            raise ValueError(
                "Every approved preview record needs a question_id"
            )
        if question_id in approved:
            raise ValueError(
                f"Duplicate question_id in preview: {question_id}"
            )
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Preview question for {question_id} must be non-empty"
            )
        claim_items = record.get("retrieval_claims")
        if not isinstance(claim_items, list):
            raise ValueError(
                f"retrieval_claims for {question_id} must be a JSON array"
            )
        claims = []
        for index, item in enumerate(claim_items, start=1):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Every claim for {question_id} must be an object"
                )
            claim = item.get("claim")
            claim_id = item.get("id")
            expected_id = f"{question_id}-C{index:02d}"
            if claim_id != expected_id:
                raise ValueError(
                    f"Claim IDs for {question_id} must be sequential; "
                    f"expected {expected_id}"
                )
            claims.append(claim)
        validated_claims = _validate_claims(claims, question_id)
        approved[question_id] = _preview_record(
            question_id, question, validated_claims
        )

    if set(approved) != set(question_ids):
        missing = [item for item in question_ids if item not in approved]
        extra = [item for item in approved if item not in question_ids]
        details = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if extra:
            details.append(f"not selected: {', '.join(extra)}")
        raise ValueError(
            "Approved preview question IDs do not match selection ("
            + "; ".join(details)
            + ")"
        )
    return [approved[question_id] for question_id in question_ids]


def _write_json_atomically(path: Path, value: Any) -> None:
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


def _apply_preview(
    dataset_path: Path,
    records: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    approved_records: list[dict[str, Any]],
    overwrite: bool,
) -> None:
    for record, approved in zip(
        selected_records, approved_records, strict=True
    ):
        if record.get("question") != approved["question"]:
            raise ValueError(
                f"Question text in preview no longer matches dataset for "
                f"{approved['question_id']}"
            )
        if "retrieval_claims" in record and not overwrite:
            raise ValueError(
                f"{approved['question_id']} already has retrieval_claims; "
                "pass --overwrite to replace them"
            )

    by_id = {record["id"]: record for record in records}
    for approved in approved_records:
        by_id[approved["question_id"]]["retrieval_claims"] = approved[
            "retrieval_claims"
        ]
    _write_json_atomically(dataset_path, records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate retrieval claims from golden answers for review, then "
            "apply an approved preview to the dataset."
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--question-ids",
        nargs="+",
        required=True,
        metavar="ID",
        help="One or more golden question IDs to process.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--preview-output",
        type=Path,
        help="Save the generated JSON preview to this path.",
    )
    parser.add_argument(
        "--apply-preview",
        type=Path,
        help="Apply claims from an edited preview without calling OpenAI.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing retrieval_claims on selected entries.",
    )
    args = parser.parse_args(argv)

    dataset_path = (
        args.dataset
        if args.dataset.is_absolute()
        else Path.cwd() / args.dataset
    )
    question_ids = args.question_ids

    try:
        if args.apply_preview is not None:
            if args.preview_output is not None:
                parser.error(
                    "--apply-preview cannot be combined with --preview-output"
                )
            records, selected_records = _load_selected_records(
                dataset_path, question_ids
            )
            preview_path = (
                args.apply_preview
                if args.apply_preview.is_absolute()
                else Path.cwd() / args.apply_preview
            )
            if preview_path.resolve() == dataset_path.resolve():
                raise ValueError(
                    "Approved preview path must not be the golden dataset path"
                )
            approved_records = _validate_preview(preview_path, question_ids)
            _apply_preview(
                dataset_path,
                records,
                selected_records,
                approved_records,
                args.overwrite,
            )
            output = {
                "status": "applied",
                "dataset": str(dataset_path),
                "records": approved_records,
            }
        else:
            if args.overwrite:
                parser.error("--overwrite is only valid with --apply-preview")
            _, selected_records = _load_selected_records(
                dataset_path, question_ids
            )
            client = _create_openai_client()
            claim_lists = [
                _request_claims(client, args.model, record)
                for record in selected_records
            ]
            output = _make_preview(args.model, selected_records, claim_lists)
            if args.preview_output is not None:
                preview_path = (
                    args.preview_output
                    if args.preview_output.is_absolute()
                    else Path.cwd() / args.preview_output
                )
                if preview_path.resolve() == dataset_path.resolve():
                    raise ValueError(
                        "Preview output path must not be the golden dataset "
                        "path"
                    )
                _write_json_atomically(preview_path, output)
                output["preview_path"] = str(preview_path)

        print(json.dumps(output, ensure_ascii=False, indent=2))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Do not expose request headers or credential-bearing client state.
        print(
            f"OpenAI claim extraction failed ({type(exc).__name__}).",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
