#!/usr/bin/env python
"""Answer golden-dataset questions using Kotaemon's configured file-QA pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import tempfile
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable


TABLES_TO_COPY = ("llm_table", "embedding", "reranking")
BATCHES = {
    "aop": ("aop.json", 0, 15),
    "shiro": ("shiro.json", 0, 15),
    "target": ("target.json", 0, 10),
    "rlpm-1": ("rlpm.json", 0, 10),
    "rlpm-2": ("rlpm.json", 10, 20),
    "rlpm-3": ("rlpm.json", 20, 30),
    "rlpm-4": ("rlpm.json", 30, 35),
    "rlpm-5": ("rlpm.json", 35, 40),
}


class _TextOnlyHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _plain_text(value: str) -> str:
    parser = _TextOnlyHTML()
    parser.feed(value)
    return " ".join(" ".join(parser.parts).split())


def _load_dataset_records(dataset: Path) -> list[dict[str, Any]]:
    try:
        records = json.loads(dataset.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read dataset {dataset}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dataset is not valid JSON: {dataset}: {exc}") from exc

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array of question records in {dataset}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected every record in {dataset} to be a JSON object")
    return records


def _validate_question_record(record: dict[str, Any], dataset: Path) -> tuple[str, str]:
    question_id = record.get("id")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError(f"A record in {dataset} has no non-empty string ID")
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Question {question_id!r} has no non-empty question text")
    return question_id, question


def _load_question(dataset: Path, question_id: str) -> dict[str, Any]:
    records = _load_dataset_records(dataset)
    matches = [
        record
        for record in records
        if record.get("id") == question_id
    ]
    if not matches:
        raise ValueError(f"Question ID {question_id!r} was not found in {dataset}")
    if len(matches) > 1:
        raise ValueError(f"Question ID {question_id!r} occurs more than once in {dataset}")

    _validate_question_record(matches[0], dataset)
    return matches[0]


def _load_batch(repo_root: Path, batch_name: str) -> list[dict[str, Any]]:
    try:
        filename, start, end = BATCHES[batch_name]
    except KeyError as exc:
        choices = ", ".join(BATCHES)
        raise ValueError(
            f"Unknown batch {batch_name!r}. Choose one of: {choices}"
        ) from exc

    dataset = repo_root / "data" / "golden" / filename
    records = _load_dataset_records(dataset)
    if len(records) < end:
        raise ValueError(
            f"Batch {batch_name!r} needs records {start + 1}–{end} from {dataset}, "
            f"but it contains only {len(records)} records"
        )
    selected_records = records[start:end]
    for record in selected_records:
        _validate_question_record(record, dataset)
    return selected_records


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _result_path(
    results_dir: Path,
    batch: str | None,
    dataset: Path | None,
    question_id: str | None,
) -> Path:
    name = batch if batch else f"{dataset.stem}_{question_id}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return results_dir / f"{safe_name}.csv"


def _decode_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _read_saved_app_config(
    database_url: str, user_id: str | None, user_management: bool
) -> tuple[dict[str, list[tuple[Any, ...]]], dict[str, Any] | None, tuple[Any, ...] | None]:
    from sqlalchemy.engine import make_url

    url = make_url(database_url)
    if not url.drivername.startswith("sqlite") or not url.database:
        raise ValueError(
            "The CLI currently needs a SQLite app database to inherit saved "
            "Kotaemon settings."
        )
    source_path = Path(url.database)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    if not source_path.is_file():
        return {}, None, None

    copied_tables: dict[str, list[tuple[Any, ...]]] = {}
    saved_settings = None
    saved_index = None
    with sqlite3.connect(source_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        for table in TABLES_TO_COPY:
            if table in table_names:
                copied_tables[table] = connection.execute(
                    f'SELECT name, spec, "default" FROM "{table}"'
                ).fetchall()

        if "settings" in table_names:
            rows = connection.execute(
                "SELECT id, user, setting FROM settings ORDER BY rowid"
            ).fetchall()
            if user_id is not None:
                saved_settings = next((row[2] for row in rows if row[1] == user_id), None)
                if saved_settings is None:
                    raise ValueError(f"No saved application settings for user {user_id!r}")
            elif not user_management:
                saved_settings = next(
                    (row[2] for row in rows if row[1] == "default"), None
                )
            elif len(rows) == 1:
                saved_settings = rows[0][2]
            elif len(rows) > 1:
                raise ValueError(
                    "Multiple users have saved settings; specify one with --user-id"
                )

        if "ktem__index" in table_names:
            row = connection.execute(
                "SELECT id, name, index_type, config FROM ktem__index "
                "WHERE name = 'File Collection' ORDER BY id LIMIT 1"
            ).fetchone()
            if row:
                saved_index = (row[0], row[1], row[2], _decode_json(row[3]))

    if saved_settings is not None:
        saved_settings = _decode_json(saved_settings)
        if not isinstance(saved_settings, dict):
            raise ValueError("Saved application settings are not a JSON object")
    return copied_tables, saved_settings, saved_index


def _set_isolated_runtime_paths(settings: Any, root: Path) -> None:
    user_data = root / "user_data"
    settings.KH_APP_DATA_DIR = root
    settings.KH_USER_DATA_DIR = user_data
    settings.KH_DATABASE = f"sqlite:///{user_data / 'sql.db'}"
    settings.KH_FILESTORAGE_PATH = str(user_data / "files")
    settings.KH_MARKDOWN_OUTPUT_DIR = root / "markdown_cache_dir"
    settings.KH_CHUNKS_OUTPUT_DIR = root / "chunks_cache_dir"
    settings.KH_ZIP_OUTPUT_DIR = root / "zip_cache_dir"
    settings.KH_ZIP_INPUT_DIR = root / "zip_cache_dir_in"

    for path in (
        user_data,
        Path(settings.KH_FILESTORAGE_PATH),
        settings.KH_MARKDOWN_OUTPUT_DIR,
        settings.KH_CHUNKS_OUTPUT_DIR,
        settings.KH_ZIP_OUTPUT_DIR,
        settings.KH_ZIP_INPUT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)

    from copy import deepcopy

    docstore = deepcopy(settings.KH_DOCSTORE)
    vectorstore = deepcopy(settings.KH_VECTORSTORE)
    if "path" in docstore:
        docstore["path"] = str(root / "docstore")
    if "path" in vectorstore:
        vectorstore["path"] = str(root / "vectorstore")
    settings.KH_DOCSTORE = docstore
    settings.KH_VECTORSTORE = vectorstore


def _seed_isolated_database(
    database_path: Path,
    copied_tables: dict[str, list[tuple[Any, ...]]],
    saved_settings: dict[str, Any] | None,
) -> None:
    # Importing these modules creates only the app configuration tables in the
    # isolated database; model managers are imported after the saved rows are copied.
    from ktem.db import models as _db_models  # noqa: F401
    from ktem.embeddings.db import EmbeddingTable  # noqa: F401
    from ktem.llms.db import LLMTable  # noqa: F401
    from ktem.rerankings.db import RerankingTable  # noqa: F401

    with sqlite3.connect(database_path) as connection:
        for table in TABLES_TO_COPY:
            for name, spec, is_default in copied_tables.get(table, []):
                connection.execute(
                    f'INSERT OR REPLACE INTO "{table}" (name, spec, "default") '
                    "VALUES (?, ?, ?)",
                    (name, json.dumps(_decode_json(spec)), is_default),
                )
        if saved_settings is not None:
            connection.execute(
                "INSERT INTO settings (id, user, setting) VALUES (?, ?, ?)",
                (uuid.uuid4().hex, "default", json.dumps(saved_settings)),
            )


def _app_default_settings(index: Any, runtime_index_id: int) -> tuple[dict, dict]:
    from ktem.components import reasonings
    from ktem.settings import (
        BaseSettingGroup,
        SettingGroup,
        SettingIndexGroup,
        SettingReasoningGroup,
    )
    from theflow.settings import settings
    from theflow.utils.modules import import_dotted_string

    reasoning_group = SettingReasoningGroup(settings=settings.SETTINGS_REASONING)
    for path in settings.KH_REASONINGS:
        reasoning_cls = import_dotted_string(path, safe=False)
        reasoning_id = reasoning_cls.get_info()["id"]
        reasonings[reasoning_id] = reasoning_cls
        reasoning_group.options[reasoning_id] = BaseSettingGroup(
            settings=reasoning_cls().get_user_settings()
        )

    index_group = SettingIndexGroup()
    index_group.options[str(runtime_index_id)] = BaseSettingGroup(
        settings=index.get_user_settings()
    )
    defaults = SettingGroup(
        application=BaseSettingGroup(settings=settings.SETTINGS_APP),
        index=index_group,
        reasoning=reasoning_group,
    )
    defaults.reasoning.finalize()
    defaults.index.finalize()
    flattened = defaults.flatten()
    return flattened, reasonings


def _get_index_definition(saved_index: tuple[Any, ...] | None, settings: Any) -> tuple:
    if saved_index is not None:
        return saved_index
    for definition in settings.KH_INDICES:
        if definition["name"] == "File Collection":
            return (
                1,
                definition["name"],
                definition["index_type"],
                definition["config"],
            )
    raise ValueError("No File Collection index is configured in flowsettings")


def _remap_saved_index_settings(
    saved_settings: dict[str, Any] | None,
    defaults: dict[str, Any],
    old_index_id: Any,
    new_index_id: int,
) -> dict[str, Any]:
    if saved_settings is None:
        return defaults
    prefix = f"index.options.{old_index_id}."
    new_prefix = f"index.options.{new_index_id}."
    remapped = {
        new_prefix + key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in saved_settings.items()
    }
    return remapped


def _prepare_question_runner(
    docs_dir: Path,
    saved_settings: dict[str, Any] | None,
    saved_index: tuple[Any, ...] | None,
) -> Callable[[str], tuple[str, str]]:
    from theflow.settings import settings
    from theflow.utils.modules import import_dotted_string

    from ktem.embeddings.manager import embedding_models_manager
    from ktem.llms.manager import llms
    from ktem.rerankings.manager import reranking_models_manager
    from ktem.index.file.index import FileIndex

    if not llms.options():
        raise ValueError("No LLM is configured. Configure one in Kotaemon Resources.")
    if not embedding_models_manager.options():
        raise ValueError(
            "No embedding model is configured. Configure one in Kotaemon Resources."
        )

    source_index_id, index_name, index_type, index_config = _get_index_definition(
        saved_index, settings
    )
    runtime_index_id = 900_000_000 + uuid.uuid4().int % 100_000_000
    index_cls = import_dotted_string(index_type, safe=False)
    if not issubclass(index_cls, FileIndex):
        raise ValueError(
            f"The configured {index_name!r} is not a file index and cannot index PDFs"
        )
    index = index_cls(
        app=None,
        id=runtime_index_id,
        name=index_name,
        config=index_config,
    )
    index.on_create()
    index.on_start()

    defaults, reasoning_registry = _app_default_settings(index, runtime_index_id)
    effective_settings = _remap_saved_index_settings(
        saved_settings, defaults, source_index_id, runtime_index_id
    )
    embedding_name = index_config.get(
        "embedding", embedding_models_manager.get_default_name()
    )
    if embedding_name not in embedding_models_manager.options():
        raise ValueError(
            f"Embedding model {embedding_name!r} from the File Collection "
            "configuration is unavailable."
        )
    if not reranking_models_manager.options():
        raise ValueError(
            "No reranker is configured. Configure one in Kotaemon Resources."
        )

    indexing_pipeline = index.get_indexing_pipeline(effective_settings, "default")
    pdfs = sorted(docs_dir.glob("*.pdf"))
    stream = indexing_pipeline.stream(pdfs, reindex=False)
    try:
        while True:
            next(stream)
    except StopIteration as stop:
        file_ids, errors, _ = stop.value
    if errors and any(errors):
        failures = [
            f"{pdfs[i].name}: {error}"
            for i, error in enumerate(errors)
            if error is not None
        ]
        raise RuntimeError("PDF indexing failed: " + "; ".join(failures))
    selected_ids = [file_id for file_id in file_ids if file_id]
    if not selected_ids:
        raise RuntimeError("The PDF indexing pipeline did not index any files.")

    reasoning_mode = effective_settings.get("reasoning.use")
    reasoning_cls = reasoning_registry.get(reasoning_mode)
    if reasoning_cls is None:
        raise ValueError(f"Unknown configured reasoning mode: {reasoning_mode!r}")

    def answer_question(question: str) -> tuple[str, str]:
        retrievers = index.get_retriever_pipelines_for_ids(
            effective_settings, "default", selected_ids
        )
        if not retrievers:
            raise RuntimeError("The configured file index did not create a retriever.")

        # Build a new reasoning pipeline for every question, matching a fresh chat.
        pipeline = reasoning_cls.get_pipeline(
            effective_settings, {"app": {}, "pipeline": {}}, retrievers
        )
        answer = ""
        citations = ""
        for response in pipeline.stream(question, uuid.uuid4().hex, []):
            if getattr(response, "channel", None) == "chat":
                answer = (
                    "" if response.content is None else answer + str(response.content)
                )
            elif getattr(response, "channel", None) == "info":
                citations = (
                    "" if response.content is None else citations + str(response.content)
                )
        return answer, citations

    return answer_question


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--batch", choices=sorted(BATCHES))
    mode.add_argument("--dataset", type=Path)
    parser.add_argument("--question-id")
    parser.add_argument(
        "--user-id",
        help="Use this app user's saved Gradio settings (required if multiple users "
        "have saved settings).",
    )
    args = parser.parse_args(argv)

    if args.batch:
        if args.question_id is not None:
            parser.error("--batch cannot be combined with --question-id")
    elif args.dataset is None or args.question_id is None:
        parser.error("single-question mode requires both --dataset and --question-id")

    dataset: Path | None = None
    try:
        if args.batch:
            questions = _load_batch(repo_root, args.batch)
        else:
            dataset = (
                args.dataset
                if args.dataset.is_absolute()
                else Path.cwd() / args.dataset
            )
            questions = [_load_question(dataset, args.question_id)]
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    docs_dir = repo_root / "data" / "raw"
    pdfs = sorted(docs_dir.glob("*.pdf")) if docs_dir.is_dir() else []
    if not pdfs:
        raise SystemExit(f"Error: no PDF files found in {docs_dir}")

    results_dir = repo_root / "results"
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Error creating results directory {results_dir}: {exc}", file=sys.stderr)
        return 1
    result_csv = _result_path(
        results_dir,
        args.batch,
        dataset,
        None if args.batch else args.question_id,
    )
    source_dataset = (
        dataset
        if dataset is not None
        else repo_root / "data" / "golden" / BATCHES[args.batch][0]
    )

    try:
        from theflow.settings import settings

        saved_tables, saved_settings, saved_index = _read_saved_app_config(
            settings.KH_DATABASE,
            args.user_id,
            getattr(settings, "KH_FEATURE_USER_MANAGEMENT", False),
        )
        with tempfile.TemporaryDirectory(prefix="kotaemon-rag-eval-") as temp_dir:
            runtime_data = Path(temp_dir)
            _set_isolated_runtime_paths(settings, runtime_data)
            db_path = Path(settings.KH_DATABASE.removeprefix("sqlite:///"))
            _seed_isolated_database(db_path, saved_tables, saved_settings)
            answer_question = _prepare_question_runner(
                docs_dir,
                saved_settings,
                saved_index,
            )
            successes = 0
            failures = 0
            result_fields = list(
                dict.fromkeys(
                    key
                    for record in questions
                    for key in record
                )
            ) + ["generated_answer", "citations", "status", "error"]
            with result_csv.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=result_fields)
                writer.writeheader()
                for record in questions:
                    question_id, question = _validate_question_record(
                        record, source_dataset
                    )
                    try:
                        answer, citations = answer_question(question)
                        generated_answer = answer or "(Sorry, I don't know)"
                        citation_text = _plain_text(citations)
                        row = {
                            **{key: _csv_value(value) for key, value in record.items()},
                            "generated_answer": generated_answer,
                            "citations": citation_text,
                            "status": "success",
                            "error": "",
                        }
                        successes += 1
                        print(f"Question ID: {question_id}")
                        print("Answer:")
                        print(generated_answer)
                        if citation_text:
                            print("Citations:")
                            print(citation_text)
                        print()
                    except Exception as exc:
                        failures += 1
                        row = {
                            **{key: _csv_value(value) for key, value in record.items()},
                            "generated_answer": "",
                            "citations": "",
                            "status": "failed",
                            "error": str(exc),
                        }
                        print(f"[{question_id}] Error: {exc}", file=sys.stderr)
                    writer.writerow(row)
                    output_file.flush()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.batch:
        print(f"Batch summary: {successes} succeeded, {failures} failed.")
    print(f"Results CSV: {result_csv}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
