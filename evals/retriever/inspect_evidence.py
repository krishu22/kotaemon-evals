#!/usr/bin/env python
"""Capture Kotaemon's selected retrieval evidence before answer generation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import html
import json
import sys
import tempfile
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Generator

import tiktoken

from scripts.run_rag_eval import (
    EvidenceCaptured,
    _load_question,
    _prepare_question_runner,
    _read_saved_app_config,
    _seed_isolated_database,
    _set_isolated_runtime_paths,
)


TOKENIZER_MODEL = "gpt-3.5-turbo"
TOKENIZER = tiktoken.encoding_for_model(TOKENIZER_MODEL)
TOKENIZER_ID = f"tiktoken/{TOKENIZER.name} ({TOKENIZER_MODEL})"


def _token_count(text: str) -> int:
    tokens = TOKENIZER.encode(
        text, allowed_special=set(), disallowed_special="all"
    )
    return len(tokens)


def _score_value(value: Any) -> int | float | None:
    if isinstance(value, Real):
        return value.item() if hasattr(value, "item") else value
    return None


def _context_candidates(doc: Any) -> list[tuple[str, str]]:
    metadata = doc.metadata
    doc_text = doc.text if isinstance(doc.text, str) else ""
    doc_type = metadata.get("type", "")
    candidates: list[tuple[str, str]] = []

    if doc_type == "table":
        table_text = metadata.get("table_origin", doc_text)
        if isinstance(table_text, str):
            candidates.append(("metadata.table_origin", table_text))
    elif doc_type == "chatbot":
        window = metadata.get("window")
        if isinstance(window, str):
            candidates.append(("metadata.window", window))
    elif doc_type == "image":
        caption = doc.get_content()
        if isinstance(caption, str):
            candidates.append(("escaped image caption", html.escape(caption)))
    elif isinstance(metadata.get("window"), str):
        candidates.append(
            ("metadata.window", metadata["window"].replace("\n", " "))
        )
    else:
        candidates.append(("text", doc_text.replace("\n", " ")))

    if doc_text and all(value != doc_text for _, value in candidates):
        candidates.append(("text", doc_text.replace("\n", " ")))
    return candidates


def _context_mapping(doc: Any, context: str) -> dict[str, Any]:
    for source, value in _context_candidates(doc):
        if not value:
            continue
        start = context.find(value)
        if start >= 0:
            return {
                "status": "full_match",
                "source": source,
                "character_span": [start, start + len(value)],
                "matched_character_count": len(value),
            }

        prefix_length = min(80, len(value))
        start = context.find(value[:prefix_length])
        if start < 0:
            continue
        low, high = prefix_length, len(value)
        while low < high:
            middle = (low + high + 1) // 2
            if context.startswith(value[:middle], start):
                low = middle
            else:
                high = middle - 1
        if low < len(value):
            return {
                "status": "partial_match",
                "source": source,
                "character_span": [start, start + low],
                "matched_character_count": low,
            }

    return {
        "status": "not_found",
        "source": None,
        "character_span": None,
        "matched_character_count": 0,
    }


def _retrieval_limits(
    retrievers: list[Any],
) -> tuple[int | list[int | None] | None, list[dict]]:
    limits = []
    requested_ks = []
    for retriever in retrievers:
        requested_k = getattr(retriever, "top_k", None)
        if not isinstance(requested_k, int):
            requested_k = None
        requested_ks.append(requested_k)

        vector_retrieval = getattr(retriever, "vector_retrieval", None)
        multiplier = getattr(vector_retrieval, "first_round_top_k_mult", None)
        candidate_k = (
            requested_k * multiplier
            if requested_k is not None and isinstance(multiplier, int)
            else None
        )
        prioritize_table = bool(getattr(retriever, "get_extra_table", False))
        limits.append(
            {
                "retriever": type(retriever).__name__,
                "requested_k": requested_k,
                "candidate_k_per_search": candidate_k,
                "final_retriever_k": requested_k,
                "additional_table_search_k": 50 if prioritize_table else None,
                "additional_table_search_enabled": prioritize_table,
            }
        )

    requested_k: int | list[int | None] | None
    if len(requested_ks) == 1:
        requested_k = requested_ks[0]
    elif requested_ks:
        requested_k = requested_ks
    else:
        requested_k = None
    return requested_k, limits


def _build_dump(
    question_id: str,
    question: str,
    docs: list[Any],
    prepared_content: Any,
    retrievers: list[Any],
    context_limit: int | None,
) -> dict[str, Any]:
    evidence_mode, final_context, images = prepared_content
    if not isinstance(final_context, str):
        raise TypeError("Kotaemon's prepared evidence context is not text")

    requested_k, retrieval_limits = _retrieval_limits(retrievers)
    chunks = []
    mappings = []
    for rank, doc in enumerate(docs, start=1):
        text = doc.text if isinstance(doc.text, str) else None
        metadata = doc.metadata
        retrieval_score = _score_value(getattr(doc, "score", None))
        reranking_scores = [
            {"score_type": key, "value": _score_value(metadata[key])}
            for key in (
                "reranking_score",
                "llm_reranking_score",
                "llm_trulens_score",
            )
            if key in metadata
        ]
        mapping = _context_mapping(doc, final_context)
        mappings.append({"rank": rank, **mapping})
        chunks.append(
            {
                "rank": rank,
                "chunk_id": getattr(doc, "doc_id", None),
                "text": text,
                "source": {
                    "filename": metadata.get("file_name"),
                    "page": metadata.get("page_label"),
                },
                "character_count": len(text) if text is not None else None,
                "token_count": (
                    _token_count(text) if text is not None else None
                ),
                "tokenizer": TOKENIZER_ID,
                "retrieval_score": (
                    {
                        "score_type": "retrieval_score",
                        "source_field": "RetrievedDocument.score",
                        "value": retrieval_score,
                    }
                    if retrieval_score is not None
                    else None
                ),
                "reranking_scores": reranking_scores or None,
                "final_context_mapping": mapping,
            }
        )

    return {
        "question_id": question_id,
        "question": question,
        "capture_stage": (
            "Ranked chunks are captured after retrieval/reranking and "
            "optional table retrieval in PrepareEvidencePipeline.run. "
            "The formatted and "
            "trimmed context is captured just before the answering pipeline; "
            "answer generation is not invoked."
        ),
        "requested_k": requested_k,
        "retrieval_limits": {
            "retrievers": retrieval_limits,
            "final_context_token_limit": context_limit,
            "actual_captured_chunk_count": len(docs),
        },
        "actual_captured_chunk_count": len(docs),
        "ranked_chunks": chunks,
        "final_context": {
            "text": final_context,
            "evidence_mode": evidence_mode,
            "images": images,
            "character_count": len(final_context),
            "token_count": _token_count(final_context),
            "tokenizer": TOKENIZER_ID,
            "chunk_mappings": mappings,
        },
    }


@contextmanager
def evidence_capture_session(
    docs_dir: Path, user_id: str | None = None
) -> Generator[Callable[[str, str], dict[str, Any]], None, None]:
    from theflow.settings import settings

    saved_tables, saved_settings, saved_index = _read_saved_app_config(
        settings.KH_DATABASE,
        user_id,
        getattr(settings, "KH_FEATURE_USER_MANAGEMENT", False),
    )
    with tempfile.TemporaryDirectory(
        prefix="kotaemon-evidence-inspect-"
    ) as temp_dir:
        runtime_data = Path(temp_dir)
        _set_isolated_runtime_paths(settings, runtime_data)
        db_path = Path(settings.KH_DATABASE.removeprefix("sqlite:///"))
        _seed_isolated_database(db_path, saved_tables, saved_settings)
        current_question: dict[str, str] = {}
        capture: dict[str, Any] = {}

        def save_evidence(
            docs: list[Any],
            prepared_content: Any,
            retrievers: list[Any],
            context_limit: int,
        ) -> None:
            capture.update(
                _build_dump(
                    current_question["id"],
                    current_question["question"],
                    docs,
                    prepared_content,
                    retrievers,
                    context_limit,
                )
            )

        answer_question = _prepare_question_runner(
            docs_dir,
            saved_settings,
            saved_index,
            evidence_callback=save_evidence,
        )

        def capture_question(question_id: str, question: str) -> dict[str, Any]:
            current_question.update(id=question_id, question=question)
            capture.clear()
            try:
                answer_question(question)
            except EvidenceCaptured:
                if not capture:
                    raise RuntimeError(
                        "Kotaemon stopped at evidence capture without producing "
                        "an evidence dump."
                    )
                return dict(capture)
            raise RuntimeError(
                "The reasoning pipeline finished without stopping at the "
                "pre-generation evidence capture point."
            )

        yield capture_question


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description=(
            "Inspect evidence selected by Kotaemon before answer generation."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=repo_root / "data" / "golden" / "rlpm.json",
    )
    parser.add_argument("--question-id", default="RLPM-39")
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "results" / "retriever" / "RLPM-39_evidence.json",
        help="Path for the JSON evidence dump.",
    )
    parser.add_argument(
        "--user-id",
        help=(
            "Use this app user's saved Gradio settings (required if multiple "
            "users have saved settings)."
        ),
    )
    args = parser.parse_args(argv)

    dataset = (
        args.dataset
        if args.dataset.is_absolute()
        else Path.cwd() / args.dataset
    )
    output_path = (
        args.output if args.output.is_absolute() else Path.cwd() / args.output
    )
    try:
        record = _load_question(dataset, args.question_id)
        question_id = record["id"]
        question = record["question"]
        docs_dir = repo_root / "data" / "raw"
        if not docs_dir.is_dir() or not any(docs_dir.glob("*.pdf")):
            raise ValueError(f"No PDF files found in {docs_dir}")

        with evidence_capture_session(docs_dir, args.user_id) as capture_question:
            capture = capture_question(question_id, question)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(capture, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    requested_k = capture["requested_k"]
    print(f"Question ID: {question_id}")
    print(f"Requested k: {requested_k}")
    print(f"Captured chunks: {capture['actual_captured_chunk_count']}")
    print(
        "Final context: "
        f"{capture['final_context']['character_count']} characters, "
        f"{capture['final_context']['token_count']} tokens"
    )
    print(f"JSON evidence: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
