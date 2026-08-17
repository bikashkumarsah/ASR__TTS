"""Configuration and bounded-memory readers for corpus-analysis inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterator

import yaml


@dataclass(frozen=True)
class CorpusSpec:
    """One locally downloaded corpus and its schema."""

    name: str
    slug: str
    kind: str
    path: str
    text_column: str = "text"
    file_glob: str = "**/*.parquet"
    metadata_columns: dict[str, str] = field(default_factory=dict)
    repo_id: str | None = None
    revision: str | None = None
    source_uri: str | None = None
    download_patterns: list[str] | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "CorpusSpec":
        required = {"name", "slug", "kind", "path"}
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Corpus entry is missing: {', '.join(sorted(missing))}")
        return cls(**{key: val for key, val in value.items() if key in cls.__dataclass_fields__})


def load_config(path: str | Path) -> list[CorpusSpec]:
    """Load and validate the corpus list from YAML."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    corpora = [CorpusSpec.from_dict(item) for item in config.get("corpora", [])]
    if not corpora:
        raise ValueError(f"No corpora configured in {path}")
    slugs = [spec.slug for spec in corpora]
    if len(slugs) != len(set(slugs)):
        raise ValueError("Corpus slugs must be unique")
    return corpora


def resolve_source_path(spec: CorpusSpec, input_root: str | Path) -> Path:
    path = Path(spec.path)
    return path if path.is_absolute() else Path(input_root) / path


def source_files(spec: CorpusSpec, input_root: str | Path) -> list[Path]:
    """Return the stable ordered set of physical files used by a corpus."""
    path = resolve_source_path(spec, input_root)
    if spec.kind == "parquet":
        return sorted(path.glob(spec.file_glob)) if path.is_dir() else [path]
    return [path]


def validate_source(spec: CorpusSpec, input_root: str | Path) -> dict:
    """Validate existence and schema without scanning the corpus."""
    files = source_files(spec, input_root)
    if not files or any(not item.is_file() for item in files):
        raise FileNotFoundError(
            f"{spec.name}: no input files found at "
            f"{resolve_source_path(spec, input_root)} (glob: {spec.file_glob})"
        )
    info = {
        "name": spec.name,
        "slug": spec.slug,
        "kind": spec.kind,
        "files": len(files),
        "bytes": sum(item.stat().st_size for item in files),
    }
    if spec.kind == "parquet":
        import pyarrow.parquet as pq

        required = {spec.text_column, *spec.metadata_columns.values()}
        missing_by_file = []
        for file_path in files:
            columns = set(pq.ParquetFile(file_path).schema.names)
            missing = sorted(required.difference(columns))
            if missing:
                missing_by_file.append(f"{file_path}: {', '.join(missing)}")
        if missing_by_file:
            raise ValueError(
                f"{spec.name}: missing configured parquet columns:\n"
                + "\n".join(missing_by_file[:10])
            )
    elif spec.kind not in {"jsonl", "text_lines", "html_paragraphs"}:
        raise ValueError(f"{spec.name}: unsupported source kind {spec.kind!r}")
    return info


def iter_records(
    spec: CorpusSpec,
    input_root: str | Path,
    *,
    read_batch_size: int = 2_048,
    max_records: int | None = None,
) -> Iterator[dict]:
    """Yield ``{"text": ..., "metadata": ...}`` records without full loading."""
    if max_records is not None and max_records <= 0:
        max_records = None
    emitted = 0
    if spec.kind == "parquet":
        iterator = _iter_parquet(spec, input_root, read_batch_size)
    elif spec.kind == "jsonl":
        iterator = _iter_jsonl(spec, input_root)
    elif spec.kind == "text_lines":
        iterator = _iter_text_lines(spec, input_root)
    elif spec.kind == "html_paragraphs":
        iterator = _iter_html_paragraphs(spec, input_root)
    else:
        raise ValueError(f"Unsupported source kind {spec.kind!r}")

    for record in iterator:
        yield record
        emitted += 1
        if max_records is not None and emitted >= max_records:
            return


def _metadata(record: dict, mapping: dict[str, str]) -> dict[str, object]:
    return {output_name: record.get(input_name) for output_name, input_name in mapping.items()}


def _iter_parquet(
    spec: CorpusSpec,
    input_root: str | Path,
    read_batch_size: int,
) -> Iterator[dict]:
    import pyarrow.parquet as pq

    columns = list(dict.fromkeys([spec.text_column, *spec.metadata_columns.values()]))
    for file_path in source_files(spec, input_root):
        parquet = pq.ParquetFile(file_path)
        for arrow_batch in parquet.iter_batches(batch_size=read_batch_size, columns=columns):
            data = arrow_batch.to_pydict()
            for index in range(arrow_batch.num_rows):
                row = {column: data[column][index] for column in columns}
                yield {
                    "text": row.get(spec.text_column),
                    "metadata": _metadata(row, spec.metadata_columns),
                }


def _iter_jsonl(spec: CorpusSpec, input_root: str | Path) -> Iterator[dict]:
    for file_path in source_files(spec, input_root):
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    yield {"text": "", "metadata": {}}
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {file_path}:{line_number}: {exc}") from exc
                yield {
                    "text": row.get(spec.text_column),
                    "metadata": _metadata(row, spec.metadata_columns),
                }


def _iter_text_lines(spec: CorpusSpec, input_root: str | Path) -> Iterator[dict]:
    for file_path in source_files(spec, input_root):
        with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                yield {"text": line.rstrip("\r\n"), "metadata": {}}


class _ParagraphParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.parts: list[str] = []
        self.ready: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "p":
            if self.depth == 0:
                self.parts = []
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "p" and self.depth:
            self.depth -= 1
            if self.depth == 0:
                self.ready.append(" ".join(self.parts))
                self.parts = []

    def handle_data(self, data: str) -> None:
        if self.depth:
            self.parts.append(data)

    def drain(self) -> list[str]:
        ready, self.ready = self.ready, []
        return ready


def _iter_html_paragraphs(spec: CorpusSpec, input_root: str | Path) -> Iterator[dict]:
    file_path = resolve_source_path(spec, input_root)
    parser = _ParagraphParser()
    paragraphs = 0
    with open(file_path, "r", encoding="utf-8", errors="replace") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            parser.feed(chunk)
            for paragraph in parser.drain():
                paragraphs += 1
                yield {"text": paragraph, "metadata": {}}
    parser.close()
    for paragraph in parser.drain():
        paragraphs += 1
        yield {"text": paragraph, "metadata": {}}
    if paragraphs == 0:
        raise ValueError(f"{file_path} contains no complete HTML <p> elements")
