#!/usr/bin/env python3
"""Fetch abstracts/full text for selected literature pool records.

Uses paper-search CLI when source-specific paper IDs exist. Unavailable records
are written to a human handoff list so the researcher can download or upload
the missing papers.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any


SELECTED_STATUSES = {"citation_pool", "seminal", "method", "recent"}


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def pool_paths(pool_dir: Path) -> dict[str, Path]:
    return {
        "pool": pool_dir / "pool.json",
        "evidence": pool_dir / "evidence",
        "logs": pool_dir / "logs",
        "contexts": pool_dir / "contexts",
    }


def load_pool(pool_dir: Path) -> dict[str, Any]:
    path = pool_paths(pool_dir)["pool"]
    if not path.exists():
        raise SystemExit(f"Pool not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_pool(pool_dir: Path, pool: dict[str, Any]) -> None:
    pool_paths(pool_dir)["pool"].write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")


def selected_records(pool: dict[str, Any], statuses: set[str], limit: int) -> list[dict[str, Any]]:
    records = [
        record
        for record in pool.get("papers", {}).values()
        if normalize_text(record.get("pool_status")) in statuses
    ]
    records.sort(key=lambda r: (normalize_text(r.get("relevance_score")), normalize_text(r.get("year"))), reverse=True)
    return records[:limit]


def base_command(args: argparse.Namespace) -> list[str]:
    if args.paper_search_repo:
        return ["uv", "run", "--directory", args.paper_search_repo, "paper-search"]
    if args.paper_search_cmd:
        return shlex.split(args.paper_search_cmd)
    return ["paper-search"]


def first_source_and_id(record: dict[str, Any]) -> tuple[str, str]:
    source = normalize_text(record.get("source")).split(";")[0].strip()
    paper_id = normalize_text(record.get("paper_id")).split(";")[0].strip()
    return source, paper_id


def safe_filename(record: dict[str, Any]) -> str:
    key = normalize_text(record.get("key")) or "paper"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", key)[:100]


def write_handoff(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Papers Requiring Human Download Or Manual Full-Text Retrieval",
        "",
        "Add local files to the pool, then update `local_pdf`, `fulltext_path`, or `abstract` in `pool.json` before drafting detailed claims.",
        "",
        "| Key | Title | DOI/URL | Reason |",
        "|---|---|---|---|",
    ]
    for row in rows:
        doi_or_url = row.get("doi") or row.get("url") or row.get("pdf_url") or ""
        title = normalize_text(row.get("title")).replace("|", "\\|")
        lines.append(f"| `{row.get('key')}` | {title} | {doi_or_url} | {row.get('_handoff_reason', '')} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch selected paper abstracts/full text from paper-search.")
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--status", default="citation_pool,seminal,method,recent")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--paper-search-cmd", default="")
    parser.add_argument("--paper-search-repo", default="")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pool_dir = Path(args.pool_dir)
    paths = pool_paths(pool_dir)
    paths["evidence"].mkdir(parents=True, exist_ok=True)
    paths["logs"].mkdir(parents=True, exist_ok=True)
    paths["contexts"].mkdir(parents=True, exist_ok=True)
    pool = load_pool(pool_dir)
    statuses = {s.strip() for s in args.status.split(",") if s.strip()}
    records = selected_records(pool, statuses, args.limit)
    base = base_command(args)
    handoff: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []

    for record in records:
        source, paper_id = first_source_and_id(record)
        if record.get("fulltext_path") and Path(record["fulltext_path"]).exists():
            log.append({"key": record.get("key"), "status": "exists", "path": record["fulltext_path"]})
            continue
        if normalize_text(record.get("abstract")):
            out_dir = paths["evidence"] / safe_filename(record)
            out_dir.mkdir(parents=True, exist_ok=True)
            abstract_path = out_dir / "abstract.md"
            abstract_path.write_text(
                f"# {record.get('title') or record.get('key')}\n\n## Abstract\n\n{record.get('abstract')}\n",
                encoding="utf-8",
            )
            record["abstract_path"] = str(abstract_path)
        if not source or not paper_id:
            if not normalize_text(record.get("abstract")):
                record["_handoff_reason"] = "missing source-specific paper ID and abstract"
                handoff.append(record)
            else:
                record["_handoff_reason"] = "abstract saved; missing source-specific paper ID for full text"
                handoff.append(record)
            continue
        out_dir = paths["evidence"] / safe_filename(record)
        out_dir.mkdir(parents=True, exist_ok=True)
        command = [*base, "read", source, paper_id, "-o", str(out_dir)]
        if args.dry_run:
            log.append({"key": record.get("key"), "status": "dry_run", "command": command})
            continue
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=args.timeout,
        )
        text = completed.stdout.strip()
        text_files = list(out_dir.glob("*.txt")) + list(out_dir.glob("*.md"))
        if text:
            text_path = out_dir / "fulltext_or_read_output.txt"
            text_path.write_text(text, encoding="utf-8")
            record["fulltext_path"] = str(text_path)
        elif text_files:
            record["fulltext_path"] = str(text_files[0])
        else:
            record["_handoff_reason"] = "paper-search read returned no text"
            handoff.append(record)
        log.append(
            {
                "key": record.get("key"),
                "status": "ok" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                "command": command,
                "stderr": completed.stderr[-1000:],
            }
        )

    handoff_path = paths["contexts"] / "needs_user_download.md"
    write_handoff(handoff_path, handoff)
    log_path = paths["logs"] / f"evidence_fetch_{now_stamp()}.json"
    log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    save_pool(pool_dir, pool)
    print(json.dumps({"records": len(records), "handoff": len(handoff), "handoff_path": str(handoff_path), "log": str(log_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
