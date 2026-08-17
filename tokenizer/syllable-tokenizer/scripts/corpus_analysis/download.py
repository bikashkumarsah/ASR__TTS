#!/usr/bin/env python3
"""Download pinned public corpus snapshots and record their exact revisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from corpus_analysis.sources import load_config, resolve_source_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(config_path: str | Path, input_root: str | Path, workers: int, skip_ieee: bool) -> Path:
    from huggingface_hub import HfApi, snapshot_download

    input_root = Path(input_root).resolve()
    input_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    entries = []
    for spec in load_config(config_path):
        target = resolve_source_path(spec, input_root)
        if spec.repo_id:
            requested_revision = spec.revision or "main"
            info = api.dataset_info(spec.repo_id, revision=requested_revision)
            target.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=spec.repo_id,
                repo_type="dataset",
                revision=info.sha,
                local_dir=target,
                allow_patterns=spec.download_patterns,
                max_workers=workers,
            )
            entries.append({
                "slug": spec.slug,
                "repo_id": spec.repo_id,
                "requested_revision": requested_revision,
                "resolved_revision": info.sha,
                "downloaded_to": str(target),
            })
        elif spec.source_uri:
            if skip_ieee:
                entries.append({
                    "slug": spec.slug,
                    "source_uri": spec.source_uri,
                    "status": "skipped by --skip-ieee",
                })
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                subprocess.run(
                    ["aws", "s3", "cp", spec.source_uri, str(target), "--no-sign-request"],
                    check=True,
                )
            entries.append({
                "slug": spec.slug,
                "source_uri": spec.source_uri,
                "downloaded_to": str(target),
                "bytes": target.stat().st_size,
                "sha256": _sha256(target),
            })
        else:
            entries.append({
                "slug": spec.slug,
                "local_path": str(target),
                "status": "not managed by downloader",
            })
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "workers": workers,
        "sources": entries,
    }
    manifest_path = input_root / "source_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download exact corpus snapshots")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--workers", type=int, default=None, help="Parallel download workers")
    parser.add_argument("--skip-ieee", action="store_true", help="Skip the public S3 compiled.txt")
    args = parser.parse_args()
    workers = args.workers or (os.cpu_count() or 1)
    if workers < 1:
        parser.error("--workers must be at least 1")
    path = download(args.config, args.input_root, workers, args.skip_ieee)
    print(f"Source revision manifest: {path}")


if __name__ == "__main__":
    main()
