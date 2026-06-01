#!/usr/bin/env python3
"""Maintain a review paper library, screening states, and evidence matrix."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any


STATUSES = {"unscreened", "include", "exclude", "maybe", "background", "seminal", "recent", "method"}


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value if item is not None)
    return str(value).strip()


def normalize_doi(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"^https?://(dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.strip().rstrip(".")


def normalize_title(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", "", text)
    return text.strip()


def dedupe_key(record: dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return "doi:" + doi
    paper_id = normalize_text(record.get("paper_id") or record.get("id")).lower()
    source = normalize_text(record.get("source")).lower()
    if source and paper_id:
        return f"id:{source}:{paper_id}"
    title = normalize_title(record.get("title"))
    authors = normalize_title(record.get("authors"))
    year = normalize_text(record.get("year"))
    return f"title:{title}|authors:{authors}|year:{year}"


def stable_key(record: dict[str, Any]) -> str:
    doi = normalize_doi(record.get("doi"))
    if doi:
        return "doi-" + re.sub(r"[^a-z0-9]+", "-", doi).strip("-")[:80]
    raw = dedupe_key(record)
    return "paper-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def empty_library() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "papers": {},
        "dedupe_index": {},
    }


def load_library(path: Path) -> dict[str, Any]:
    if not path.exists():
        return empty_library()
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("papers", {})
    data.setdefault("dedupe_index", {})
    return data


def save_library(path: Path, library: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    library["updated_at"] = now_iso()
    path.write_text(json.dumps(library, ensure_ascii=False, indent=2), encoding="utf-8")


def canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "key": normalize_text(row.get("key")),
        "title": normalize_text(row.get("title")),
        "authors": normalize_text(row.get("authors")),
        "year": normalize_text(row.get("year") or row.get("publication_year") or row.get("published")),
        "source": normalize_text(row.get("source")),
        "paper_id": normalize_text(row.get("paper_id") or row.get("id")),
        "doi": normalize_doi(row.get("doi")),
        "url": normalize_text(row.get("url") or row.get("paper_url") or row.get("landing_page_url")),
        "pdf_url": normalize_text(row.get("pdf_url") or row.get("open_access_pdf")),
        "abstract": normalize_text(row.get("abstract") or row.get("summary")),
        "recall_queries": normalize_text(row.get("recall_queries") or row.get("_recall_query")),
        "screening_status": normalize_text(row.get("screening_status")) or "unscreened",
        "screening_reason": normalize_text(row.get("screening_reason")),
        "claim_supported": normalize_text(row.get("claim_supported")),
        "limitations": normalize_text(row.get("limitations")),
        "use_in_review": normalize_text(row.get("use_in_review")),
        "notes": normalize_text(row.get("notes")),
        "imported_at": normalize_text(row.get("imported_at")) or now_iso(),
    }


def merge_record(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for field, value in incoming.items():
        if field in {"source", "recall_queries"} and value:
            parts = {
                p.strip()
                for p in (normalize_text(existing.get(field)) + ";" + normalize_text(value)).split(";")
                if p.strip()
            }
            merged[field] = "; ".join(sorted(parts))
        elif not normalize_text(merged.get(field)) and normalize_text(value):
            merged[field] = value
    return merged


def import_rows(library: dict[str, Any], rows: list[dict[str, Any]]) -> tuple[int, int]:
    added = 0
    updated = 0
    for row in rows:
        record = canonical_record(row)
        dkey = dedupe_key(record)
        key = library["dedupe_index"].get(dkey) or record.get("key") or stable_key(record)
        record["key"] = key
        if key in library["papers"]:
            library["papers"][key] = merge_record(library["papers"][key], record)
            updated += 1
        else:
            library["papers"][key] = record
            added += 1
        library["dedupe_index"][dkey] = key
    return added, updated


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def markdown_escape(text: str) -> str:
    return normalize_text(text).replace("|", "\\|").replace("\n", " ")


def included_records(library: dict[str, Any], include_background: bool = False) -> list[dict[str, Any]]:
    allowed = {"include", "seminal", "recent", "method"}
    if include_background:
        allowed.add("background")
    return [
        record
        for record in library["papers"].values()
        if normalize_text(record.get("screening_status")) in allowed
    ]


def cmd_init(args: argparse.Namespace) -> int:
    path = Path(args.library)
    if path.exists() and not args.force:
        raise SystemExit(f"Library exists: {path}. Use --force to overwrite.")
    save_library(path, empty_library())
    print(json.dumps({"library": str(path), "status": "created"}, ensure_ascii=False))
    return 0


def cmd_import_recall(args: argparse.Namespace) -> int:
    library_path = Path(args.library)
    library = load_library(library_path)
    recall_dir = Path(args.recall_dir)
    candidates = [recall_dir / "papers.jsonl", recall_dir / "screening.csv", recall_dir / "papers.csv"]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.exists():
            rows = read_jsonl(candidate) if candidate.suffix == ".jsonl" else read_csv(candidate)
            break
    if not rows:
        raise SystemExit(f"No recall rows found in {recall_dir}")
    added, updated = import_rows(library, rows)
    save_library(library_path, library)
    print(json.dumps({"library": str(library_path), "added": added, "updated": updated, "total": len(library["papers"])}, ensure_ascii=False))
    return 0


def cmd_import_csv(args: argparse.Namespace) -> int:
    library_path = Path(args.library)
    library = load_library(library_path)
    rows = read_csv(Path(args.csv))
    added, updated = import_rows(library, rows)
    save_library(library_path, library)
    print(json.dumps({"library": str(library_path), "added": added, "updated": updated, "total": len(library["papers"])}, ensure_ascii=False))
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    status = args.status.strip().lower()
    if status not in STATUSES:
        raise SystemExit(f"Unknown status: {status}. Allowed: {', '.join(sorted(STATUSES))}")
    library_path = Path(args.library)
    library = load_library(library_path)
    record = library["papers"].get(args.key)
    if not record:
        raise SystemExit(f"Paper key not found: {args.key}")
    record["screening_status"] = status
    record["screening_reason"] = args.reason or record.get("screening_reason", "")
    if args.notes:
        record["notes"] = args.notes
    save_library(library_path, library)
    print(json.dumps({"key": args.key, "status": status}, ensure_ascii=False))
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    library_path = Path(args.library)
    library = load_library(library_path)
    record = library["papers"].get(args.key)
    if not record:
        raise SystemExit(f"Paper key not found: {args.key}")
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"Invalid --set value: {item}. Use field=value.")
        field, value = item.split("=", 1)
        record[field.strip()] = value.strip()
    save_library(library_path, library)
    print(json.dumps({"key": args.key, "updated": len(args.set)}, ensure_ascii=False))
    return 0


def cmd_export_matrix(args: argparse.Namespace) -> int:
    library = load_library(Path(args.library))
    records = included_records(library, include_background=args.include_background)
    if args.all:
        records = list(library["papers"].values())
    records.sort(key=lambda r: (normalize_text(r.get("year")), normalize_text(r.get("title"))), reverse=True)
    lines = [
        "| Key | Title | Year | Status | Source | DOI/URL | Claim Supported | Limitation | Use In Review |",
        "|---|---|---:|---|---|---|---|---|---|",
    ]
    for record in records:
        doi_or_url = record.get("doi") or record.get("url") or record.get("pdf_url")
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(value)
                for value in [
                    record.get("key", ""),
                    record.get("title", ""),
                    record.get("year", ""),
                    record.get("screening_status", ""),
                    record.get("source", ""),
                    doi_or_url,
                    record.get("claim_supported", ""),
                    record.get("limitations", ""),
                    record.get("use_in_review", ""),
                ]
            )
            + " |"
        )
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "records": len(records)}, ensure_ascii=False))
    return 0


def cmd_export_csv(args: argparse.Namespace) -> int:
    library = load_library(Path(args.library))
    records = list(library["papers"].values())
    fields = [
        "key",
        "screening_status",
        "screening_reason",
        "title",
        "authors",
        "year",
        "source",
        "doi",
        "url",
        "paper_id",
        "claim_supported",
        "limitations",
        "use_in_review",
        "notes",
    ]
    with Path(args.out).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
    print(json.dumps({"out": args.out, "records": len(records)}, ensure_ascii=False))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    library = load_library(Path(args.library))
    records = list(library["papers"].values())
    counts: dict[str, int] = {status: 0 for status in sorted(STATUSES)}
    missing_doi_or_url = []
    missing_evidence = []
    unscreened = []
    duplicates: dict[str, list[str]] = {}
    for record in records:
        status = normalize_text(record.get("screening_status")) or "unscreened"
        counts[status] = counts.get(status, 0) + 1
        if not (record.get("doi") or record.get("url") or record.get("pdf_url")):
            missing_doi_or_url.append(record)
        if status in {"include", "seminal", "recent", "method"} and not record.get("claim_supported"):
            missing_evidence.append(record)
        if status == "unscreened":
            unscreened.append(record)
        dkey = dedupe_key(record)
        duplicates.setdefault(dkey, []).append(record.get("key", ""))
    duplicate_groups = [keys for keys in duplicates.values() if len(keys) > 1]
    lines = [
        "# Review Library Audit",
        "",
        f"- Total records: {len(records)}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Blocking Checks",
            "",
            f"- Unscreened records: {len(unscreened)}",
            f"- Included records missing `claim_supported`: {len(missing_evidence)}",
            f"- Records missing DOI/URL/PDF URL: {len(missing_doi_or_url)}",
            f"- Duplicate-key groups: {len(duplicate_groups)}",
        ]
    )
    if missing_evidence:
        lines.extend(["", "## Included Records Missing Evidence Notes", ""])
        for record in missing_evidence[:50]:
            lines.append(f"- `{record.get('key')}` {record.get('title')}")
    if unscreened:
        lines.extend(["", "## First Unscreened Records", ""])
        for record in unscreened[:50]:
            lines.append(f"- `{record.get('key')}` {record.get('title')}")
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "records": len(records), "unscreened": len(unscreened)}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage a top-journal review literature library.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create an empty library JSON.")
    p_init.add_argument("--library", required=True)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_import = sub.add_parser("import-recall", help="Import outputs from paper_recall.py.")
    p_import.add_argument("--library", required=True)
    p_import.add_argument("--recall-dir", required=True)
    p_import.set_defaults(func=cmd_import_recall)

    p_import_csv = sub.add_parser("import-csv", help="Import a CSV with title/year/doi/source fields.")
    p_import_csv.add_argument("--library", required=True)
    p_import_csv.add_argument("--csv", required=True)
    p_import_csv.set_defaults(func=cmd_import_csv)

    p_screen = sub.add_parser("screen", help="Set screening status for a paper key.")
    p_screen.add_argument("--library", required=True)
    p_screen.add_argument("--key", required=True)
    p_screen.add_argument("--status", required=True)
    p_screen.add_argument("--reason", default="")
    p_screen.add_argument("--notes", default="")
    p_screen.set_defaults(func=cmd_screen)

    p_update = sub.add_parser("update", help="Set evidence fields with --set field=value.")
    p_update.add_argument("--library", required=True)
    p_update.add_argument("--key", required=True)
    p_update.add_argument("--set", action="append", default=[], required=True)
    p_update.set_defaults(func=cmd_update)

    p_matrix = sub.add_parser("export-matrix", help="Export an evidence matrix markdown table.")
    p_matrix.add_argument("--library", required=True)
    p_matrix.add_argument("--out", required=True)
    p_matrix.add_argument("--all", action="store_true")
    p_matrix.add_argument("--include-background", action="store_true")
    p_matrix.set_defaults(func=cmd_export_matrix)

    p_csv = sub.add_parser("export-csv", help="Export all library records to CSV.")
    p_csv.add_argument("--library", required=True)
    p_csv.add_argument("--out", required=True)
    p_csv.set_defaults(func=cmd_export_csv)

    p_audit = sub.add_parser("audit", help="Write a basic library audit report.")
    p_audit.add_argument("--library", required=True)
    p_audit.add_argument("--out", required=True)
    p_audit.set_defaults(func=cmd_audit)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
