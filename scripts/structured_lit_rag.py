#!/usr/bin/env python3
"""Build a small TreeSearch-inspired literature RAG index.

This is a no-dependency fallback inspired by shibing624/TreeSearch: preserve
document headings, index nodes into SQLite FTS5, and retrieve structured nodes
instead of arbitrary fixed chunks. If pytreesearch is installed, agents may use
TreeSearch directly; this script keeps the skill runnable without extra setup.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def pool_paths(pool_dir: Path) -> dict[str, Path]:
    return {
        "pool": pool_dir / "pool.json",
        "cards": pool_dir / "cards",
        "evidence": pool_dir / "evidence",
        "indexes": pool_dir / "indexes",
    }


def parse_markdown_nodes(path: Path, text: str) -> list[dict[str, str]]:
    nodes: list[dict[str, str]] = []
    current_title = path.stem
    current_path = [path.stem]
    current_lines: list[str] = []

    def flush() -> None:
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                nodes.append(
                    {
                        "doc_path": str(path),
                        "node_path": " > ".join(current_path),
                        "title": current_title,
                        "body": body,
                    }
                )

    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            current_path[:] = current_path[:level]
            while len(current_path) < level:
                current_path.append("")
            current_path[level - 1] = title
            del current_path[level:]
            current_title = title
            current_lines.clear()
        else:
            current_lines.append(line)
    flush()
    return nodes


def parse_plain_nodes(path: Path, text: str) -> list[dict[str, str]]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    nodes = []
    for idx, para in enumerate(paras, 1):
        title = para.splitlines()[0][:100]
        nodes.append({"doc_path": str(path), "node_path": f"{path.stem} > p{idx}", "title": title, "body": para})
    return nodes


def discover_files(pool_dir: Path) -> list[Path]:
    paths = pool_paths(pool_dir)
    files: list[Path] = []
    for root in [paths["cards"], paths["evidence"]]:
        if root.exists():
            files.extend(p for p in root.rglob("*") if p.suffix.lower() in {".md", ".txt"} and p.is_file())
    pool_path = paths["pool"]
    if pool_path.exists():
        pool = json.loads(pool_path.read_text(encoding="utf-8"))
        abstracts_dir = paths["indexes"] / "abstract_nodes"
        abstracts_dir.mkdir(parents=True, exist_ok=True)
        for record in pool.get("papers", {}).values():
            for field in ("fulltext_text_path", "fulltext_path", "abstract_path"):
                saved = normalize_text(record.get(field))
                if saved:
                    saved_path = Path(saved)
                    if saved_path.exists() and saved_path.suffix.lower() in {".md", ".txt"}:
                        files.append(saved_path)
            abstract = normalize_text(record.get("abstract"))
            if abstract:
                key = re.sub(r"[^A-Za-z0-9_.-]+", "-", normalize_text(record.get("key")) or "paper")[:100]
                p = abstracts_dir / f"{key}.md"
                title = normalize_text(record.get("title")) or key
                p.write_text(f"# {title}\n\n## Abstract\n\n{abstract}\n", encoding="utf-8")
                files.append(p)
    return sorted(set(files))


def create_schema(conn: sqlite3.Connection) -> None:
    conn.execute("DROP TABLE IF EXISTS nodes")
    conn.execute("DROP TABLE IF EXISTS nodes_fts")
    conn.execute(
        "CREATE TABLE nodes (id INTEGER PRIMARY KEY, doc_path TEXT, node_path TEXT, title TEXT, body TEXT)"
    )
    conn.execute(
        "CREATE VIRTUAL TABLE nodes_fts USING fts5(title, body, node_path, content='nodes', content_rowid='id')"
    )


def fts_query(query: str) -> str:
    terms = re.findall(r"[\w\u4e00-\u9fff]+", query)
    return " OR ".join(terms) if terms else query


def cmd_build(args: argparse.Namespace) -> int:
    pool_dir = Path(args.pool_dir)
    index_dir = pool_paths(pool_dir)["indexes"]
    index_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path(args.db) if args.db else index_dir / "lit_rag.sqlite"
    conn = sqlite3.connect(db_path)
    create_schema(conn)
    count = 0
    for path in discover_files(pool_dir):
        text = read_text(path)
        nodes = parse_markdown_nodes(path, text) if path.suffix.lower() == ".md" else parse_plain_nodes(path, text)
        for node in nodes:
            cursor = conn.execute(
                "INSERT INTO nodes (doc_path, node_path, title, body) VALUES (?, ?, ?, ?)",
                (node["doc_path"], node["node_path"], node["title"], node["body"]),
            )
            rowid = cursor.lastrowid
            conn.execute(
                "INSERT INTO nodes_fts (rowid, title, body, node_path) VALUES (?, ?, ?, ?)",
                (rowid, node["title"], node["body"], node["node_path"]),
            )
            count += 1
    conn.commit()
    conn.close()
    print(json.dumps({"db": str(db_path), "nodes": count}, ensure_ascii=False))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    conn = sqlite3.connect(db_path)
    expression = args.fts_expression or fts_query(args.query)
    rows = conn.execute(
        """
        SELECT nodes.doc_path, nodes.node_path, nodes.title, snippet(nodes_fts, 1, '[', ']', ' ... ', 24) AS snippet,
               bm25(nodes_fts) AS score
        FROM nodes_fts
        JOIN nodes ON nodes.id = nodes_fts.rowid
        WHERE nodes_fts MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (expression, args.top_k),
    ).fetchall()
    conn.close()
    result = [
        {"doc_path": row[0], "node_path": row[1], "title": row[2], "snippet": row[3], "score": row[4]}
        for row in rows
    ]
    if args.out:
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build/search a structure-aware literature RAG index.")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--pool-dir", required=True)
    p_build.add_argument("--db", default="")
    p_build.set_defaults(func=cmd_build)
    p_search = sub.add_parser("search")
    p_search.add_argument("--db", required=True)
    p_search.add_argument("--query", required=True)
    p_search.add_argument("--top-k", type=int, default=10)
    p_search.add_argument("--fts-expression", default="")
    p_search.add_argument("--out", default="")
    p_search.set_defaults(func=cmd_search)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
