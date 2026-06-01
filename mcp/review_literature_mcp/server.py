#!/usr/bin/env python3
"""MCP wrapper for draft-first literature retrieval and verification.

This prototype intentionally wraps the bundled skill scripts instead of
reimplementing their internals. It keeps the MCP surface stable while the
deterministic backends continue to evolve inside the skill.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - runtime setup hint
    raise SystemExit("Install MCP first: pip install mcp") from exc


SKILL_DIR = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = SKILL_DIR / "scripts"
DEFAULT_TIMEOUT = 3600

mcp = FastMCP("review-literature-mcp")


def script_path(name: str) -> Path:
    path = SCRIPTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing bundled script: {path}")
    return path


def run_script(script: str, args: list[str], cwd: str | None = None, timeout: int = DEFAULT_TIMEOUT) -> dict[str, Any]:
    command = [sys.executable, str(script_path(script)), *[str(item) for item in args if str(item) != ""]]
    proc = subprocess.run(command, cwd=cwd or None, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    parsed: Any = None
    stdout = proc.stdout.strip()
    if stdout:
        last_line = stdout.splitlines()[-1]
        try:
            parsed = json.loads(last_line)
        except json.JSONDecodeError:
            parsed = None
    return {
        "ok": proc.returncode == 0,
        "returncode": proc.returncode,
        "command": command,
        "cwd": cwd or "",
        "json": parsed,
        "stdout_tail": "\n".join(stdout.splitlines()[-20:]),
        "stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-20:]),
    }


@mcp.tool()
def extract_draft_citation_assets(project_dir: str, draft_dir: str, out_dir: str, topic: str = "") -> dict[str, Any]:
    """Extract explicit references, in-text markers, and candidate paper clues from drafts."""
    return run_script(
        "draft_citation_assets.py",
        ["build", "--draft-dir", draft_dir, "--out-dir", out_dir, "--topic", topic],
        cwd=project_dir,
    )


@mcp.tool()
def ingest_user_pubmed_set(
    project_dir: str,
    input_files: list[str],
    out_dir: str,
    topic: str = "",
    set_name: str = "user_pubmed_set",
    packet_size: int = 40,
    min_screening_score: int = 3,
    max_screening_candidates: int = 0,
    include_quarantine: bool = False,
) -> dict[str, Any]:
    """Parse user-supplied PubMed Summary/Abstract exports into a screened seed lane."""
    args = ["parse", "--out-dir", out_dir, "--topic", topic, "--set-name", set_name, "--packet-size", str(packet_size), "--min-screening-score", str(min_screening_score)]
    for path in input_files:
        args.extend(["--input", path])
    if max_screening_candidates:
        args.extend(["--max-screening-candidates", str(max_screening_candidates)])
    if include_quarantine:
        args.append("--include-quarantine")
    return run_script("pubmed_user_set_ingestor.py", args, cwd=project_dir)


@mcp.tool()
def collect_user_pubmed_screening(
    project_dir: str,
    set_dir: str,
    screening_csvs: list[str] | None = None,
    out_dir: str = "",
    verification_out_dir: str = "",
) -> dict[str, Any]:
    """Collect user PubMed seed screening decisions and create verifier-ready rows."""
    args = ["collect-screening", "--set-dir", set_dir]
    for path in screening_csvs or []:
        args.extend(["--screening-csv", path])
    if out_dir:
        args.extend(["--out-dir", out_dir])
    if verification_out_dir:
        args.extend(["--verification-out-dir", verification_out_dir])
    return run_script("pubmed_user_set_ingestor.py", args, cwd=project_dir)


@mcp.tool()
def build_literature_candidate_board(
    project_dir: str,
    out_dir: str = "",
    include_path_contains: list[str] | None = None,
    exclude_path_contains: list[str] | None = None,
) -> dict[str, Any]:
    """Build a read-only chief-editor board across all candidate, screening, and verification lanes."""
    args = ["build", "--project-dir", project_dir]
    if out_dir:
        args.extend(["--out-dir", out_dir])
    for marker in include_path_contains or []:
        args.extend(["--include-path-contains", marker])
    for marker in exclude_path_contains or []:
        args.extend(["--exclude-path-contains", marker])
    return run_script("literature_candidate_board.py", args, cwd=project_dir)


@mcp.tool()
def build_candidate_board_agent_packets(
    project_dir: str,
    board_dir: str = "./review-data/02_literature/candidate_board",
    out_dir: str = "",
    packet_size: int = 25,
    max_screening_items: int = 0,
    max_verification_items: int = 0,
) -> dict[str, Any]:
    """Create Codex-subtask packets from candidate-board screening and verification worklists."""
    args = [
        "packets",
        "--project-dir",
        project_dir,
        "--board-dir",
        board_dir,
        "--packet-size",
        str(packet_size),
        "--max-screening-items",
        str(max_screening_items),
        "--max-verification-items",
        str(max_verification_items),
    ]
    if out_dir:
        args.extend(["--out-dir", out_dir])
    return run_script("literature_candidate_board.py", args, cwd=project_dir)


@mcp.tool()
def collect_candidate_board_agent_decisions(
    project_dir: str,
    board_dir: str = "./review-data/02_literature/candidate_board",
    input_dirs: list[str] | None = None,
    input_csvs: list[str] | None = None,
    out_dir: str = "",
) -> dict[str, Any]:
    """Collect candidate-board subagent decisions into verifier, repair, full-text, and human-decision queues."""
    args = ["collect", "--project-dir", project_dir, "--board-dir", board_dir]
    for path in input_dirs or []:
        args.extend(["--input-dir", path])
    for path in input_csvs or []:
        args.extend(["--input-csv", path])
    if out_dir:
        args.extend(["--out-dir", out_dir])
    return run_script("literature_candidate_board.py", args, cwd=project_dir)


@mcp.tool()
def build_hidden_candidate_packets(project_dir: str, draft_dir: str, out_dir: str, topic: str = "", max_chunks: int = 0) -> dict[str, Any]:
    """Create Codex-subtask packets for hidden paper-candidate mining from draft prose."""
    args = ["packet", "--draft-dir", draft_dir, "--out-dir", out_dir, "--topic", topic]
    if max_chunks:
        args.extend(["--max-chunks", str(max_chunks)])
    return run_script("draft_literature_candidate_miner.py", args, cwd=project_dir)


@mcp.tool()
def build_literature_discovery_packets(
    project_dir: str,
    draft_dir: str,
    draft_assets_dir: str,
    out_dir: str,
    topic: str = "",
    max_packets: int = 0,
) -> dict[str, Any]:
    """Create chief-Codex and subagent packets for draft-first literature discovery."""
    args = ["plan", "--draft-dir", draft_dir, "--draft-assets-dir", draft_assets_dir, "--out-dir", out_dir, "--topic", topic]
    if max_packets:
        args.extend(["--max-packets", str(max_packets)])
    return run_script("draft_literature_discovery_orchestrator.py", args, cwd=project_dir)


@mcp.tool()
def collect_literature_discovery_outputs(
    project_dir: str,
    input_path: str,
    out_dir: str,
    is_dir: bool = False,
    reference_candidates_csv: str = "",
) -> dict[str, Any]:
    """Collect literature-discovery subagent outputs into verifier-ready candidates and API tasks."""
    arg_name = "--input-dir" if is_dir else "--input"
    args = ["collect", arg_name, input_path, "--out-dir", out_dir]
    if reference_candidates_csv:
        args.extend(["--reference-candidates-csv", reference_candidates_csv])
    return run_script("draft_literature_discovery_orchestrator.py", args, cwd=project_dir)


@mcp.tool()
def collect_hidden_candidate_outputs(
    project_dir: str,
    input_path: str,
    out_dir: str,
    is_dir: bool = False,
    reference_candidates_csv: str = "",
) -> dict[str, Any]:
    """Collect Codex-subtask outputs and mark overlap with draft-native reference assets."""
    arg_name = "--input-dir" if is_dir else "--input"
    args = ["collect", arg_name, input_path, "--out-dir", out_dir]
    if reference_candidates_csv:
        args.extend(["--reference-candidates-csv", reference_candidates_csv])
    return run_script("draft_literature_candidate_miner.py", args, cwd=project_dir)


@mcp.tool()
def build_identity_normalization_packets(project_dir: str, candidate_csv: str, out_dir: str, topic: str = "", batch_size: int = 12) -> dict[str, Any]:
    """Create Codex-subtask packets to map informal system names to official paper identities."""
    return run_script(
        "draft_literature_candidate_miner.py",
        ["normalize-packet", "--candidate-csv", candidate_csv, "--out-dir", out_dir, "--topic", topic, "--batch-size", str(batch_size)],
        cwd=project_dir,
    )


@mcp.tool()
def merge_normalized_identities(project_dir: str, candidate_csv: str, normalized_csv: str, out_dir: str, topic: str = "") -> dict[str, Any]:
    """Merge normalized paper identities into deterministic-verifier input rows."""
    return run_script(
        "draft_literature_candidate_miner.py",
        ["merge-normalized", "--candidate-csv", candidate_csv, "--normalized-csv", normalized_csv, "--out-dir", out_dir, "--topic", topic],
        cwd=project_dir,
    )


@mcp.tool()
def pubmed_abstracts_to_verifier(
    project_dir: str,
    abstracts_csv: str,
    out_dir: str,
    tasks_csv: str = "",
    include_status: str = "",
) -> dict[str, Any]:
    """Convert screened PubMed abstract hits into deterministic-verifier candidate rows."""
    args = ["pubmed-to-verifier", "--abstracts-csv", abstracts_csv, "--out-dir", out_dir]
    if tasks_csv:
        args.extend(["--tasks-csv", tasks_csv])
    if include_status:
        args.extend(["--include-status", include_status])
    return run_script("draft_literature_candidate_miner.py", args, cwd=project_dir)


@mcp.tool()
def plan_supplemental_recall_screening(
    project_dir: str,
    topic: str,
    draft_assets_dir: str,
    literature_discovery_dir: str,
    out_dir: str,
    max_tasks: int = 160,
) -> dict[str, Any]:
    """Create broad-but-bounded supplemental recall tasks after framework confirmation."""
    return run_script(
        "supplemental_recall_screening.py",
        [
            "plan",
            "--topic",
            topic,
            "--draft-assets-dir",
            draft_assets_dir,
            "--literature-discovery-dir",
            literature_discovery_dir,
            "--out-dir",
            out_dir,
            "--max-tasks",
            str(max_tasks),
        ],
        cwd=project_dir,
    )


@mcp.tool()
def run_supplemental_pubmed_recall(
    project_dir: str,
    tasks_csv: str,
    out_dir: str,
    max_tasks: int = 0,
    max_results: int = 20,
    workers: int = 2,
    email: str = "",
    resume: bool = True,
) -> dict[str, Any]:
    """Run PubMed recall for supplemental recall tasks."""
    args = [
        "pubmed",
        "--tasks-csv",
        tasks_csv,
        "--out-dir",
        out_dir,
        "--max-results",
        str(max_results),
        "--workers",
        str(workers),
    ]
    if max_tasks:
        args.extend(["--max-tasks", str(max_tasks)])
    if email:
        args.extend(["--email", email])
    if resume:
        args.append("--resume")
    return run_script("supplemental_recall_screening.py", args, cwd=project_dir, timeout=DEFAULT_TIMEOUT)


@mcp.tool()
def collect_supplemental_recall_candidates(
    project_dir: str,
    out_dir: str,
    topic: str = "",
    candidate_csvs: list[str] | None = None,
    seed_csvs: list[str] | None = None,
    recall_dirs: list[str] | None = None,
    packet_size: int = 50,
    min_screening_score: int = -999,
    max_screening_candidates: int = 0,
    include_quarantine: bool = False,
) -> dict[str, Any]:
    """Deduplicate supplemental recall candidates, seed-lock curated assets, and create ranked screening packets."""
    args = ["collect-candidates", "--out-dir", out_dir]
    if topic:
        args.extend(["--topic", topic])
    for path in candidate_csvs or []:
        args.extend(["--candidate-csv", path])
    for path in seed_csvs or []:
        args.extend(["--seed-csv", path])
    for path in recall_dirs or []:
        args.extend(["--recall-dir", path])
    args.extend(["--packet-size", str(packet_size), "--min-screening-score", str(min_screening_score), "--max-screening-candidates", str(max_screening_candidates)])
    if include_quarantine:
        args.append("--include-quarantine")
    return run_script("supplemental_recall_screening.py", args, cwd=project_dir)


@mcp.tool()
def collect_supplemental_screening_decisions(
    project_dir: str,
    candidates_csv: str,
    screening_csv: str,
    out_dir: str,
    screening_csvs: list[str] | None = None,
) -> dict[str, Any]:
    """Collect agent/human supplemental screening decisions into verifier-ready rows."""
    args = ["collect-screening", "--candidates-csv", candidates_csv, "--out-dir", out_dir]
    paths = screening_csvs or [screening_csv]
    for path in paths:
        args.extend(["--screening-csv", path])
    return run_script(
        "supplemental_recall_screening.py",
        args,
        cwd=project_dir,
    )


@mcp.tool()
def audit_supplemental_recall_pool(
    project_dir: str,
    collection_dir: str,
    screened_dir: str = "",
    verification_dir: str = "",
    out_dir: str = "",
    top_n: int = 40,
) -> dict[str, Any]:
    """Audit broad recall pool formation, screened coverage, backlog, and quarantine rescue candidates."""
    args = ["audit-pool", "--collection-dir", collection_dir, "--top-n", str(top_n)]
    if screened_dir:
        args.extend(["--screened-dir", screened_dir])
    if verification_dir:
        args.extend(["--verification-dir", verification_dir])
    if out_dir:
        args.extend(["--out-dir", out_dir])
    return run_script("supplemental_recall_screening.py", args, cwd=project_dir)


@mcp.tool()
def build_recall_supervision_packet(
    project_dir: str,
    collection_dir: str,
    topic: str = "",
    screened_dir: str = "",
    out_dir: str = "",
    backlog_n: int = 80,
    deferred_n: int = 80,
    quarantine_n: int = 40,
    max_pubmed_tasks: int = 60,
    deep_dive_task_quota: int = 0,
    formal_version_task_quota: int = 0,
    framework_task_quota: int = 0,
    governance_task_quota: int = 0,
    rescue_task_quota: int = 0,
    strategic_pubmed_tasks: int = 12,
) -> dict[str, Any]:
    """Create Codex/subagent recall-supervision packets and bounded PubMed deep-dive tasks."""
    args = [
        "supervise-pool",
        "--collection-dir",
        collection_dir,
        "--backlog-n",
        str(backlog_n),
        "--deferred-n",
        str(deferred_n),
        "--quarantine-n",
        str(quarantine_n),
        "--max-pubmed-tasks",
        str(max_pubmed_tasks),
        "--strategic-pubmed-tasks",
        str(strategic_pubmed_tasks),
    ]
    quota_args = [
        ("--deep-dive-task-quota", deep_dive_task_quota),
        ("--formal-version-task-quota", formal_version_task_quota),
        ("--framework-task-quota", framework_task_quota),
        ("--governance-task-quota", governance_task_quota),
        ("--rescue-task-quota", rescue_task_quota),
    ]
    for flag, value in quota_args:
        if value:
            args.extend([flag, str(value)])
    if topic:
        args.extend(["--topic", topic])
    if screened_dir:
        args.extend(["--screened-dir", screened_dir])
    if out_dir:
        args.extend(["--out-dir", out_dir])
    return run_script("supplemental_recall_screening.py", args, cwd=project_dir)


@mcp.tool()
def collect_recall_supervision_decisions(
    project_dir: str,
    out_dir: str,
    supervision_csv: str,
    topic: str = "",
    candidates_csv: str = "",
    supervision_csvs: list[str] | None = None,
    max_pubmed_tasks: int = 80,
    deep_dive_task_quota: int = 0,
    formal_version_task_quota: int = 0,
    framework_task_quota: int = 0,
    governance_task_quota: int = 0,
    rescue_task_quota: int = 0,
    strategic_pubmed_tasks: int = 12,
) -> dict[str, Any]:
    """Collect recall-supervision decisions into PubMed tasks and rescue candidates."""
    args = [
        "collect-supervision",
        "--out-dir",
        out_dir,
        "--max-pubmed-tasks",
        str(max_pubmed_tasks),
        "--strategic-pubmed-tasks",
        str(strategic_pubmed_tasks),
    ]
    quota_args = [
        ("--deep-dive-task-quota", deep_dive_task_quota),
        ("--formal-version-task-quota", formal_version_task_quota),
        ("--framework-task-quota", framework_task_quota),
        ("--governance-task-quota", governance_task_quota),
        ("--rescue-task-quota", rescue_task_quota),
    ]
    for flag, value in quota_args:
        if value:
            args.extend([flag, str(value)])
    paths = supervision_csvs or [supervision_csv]
    for path in paths:
        args.extend(["--supervision-csv", path])
    if topic:
        args.extend(["--topic", topic])
    if candidates_csv:
        args.extend(["--candidates-csv", candidates_csv])
    return run_script("supplemental_recall_screening.py", args, cwd=project_dir)


@mcp.tool()
def verify_paper_candidates(project_dir: str, candidate_csv: str, out_dir: str, workers: int = 4, timeout: int = 40, resume: bool = True) -> dict[str, Any]:
    """Verify candidates through PubMed, Crossref, OpenAlex, OpenReview, and official pages."""
    args = [
        "verify",
        "--candidate-csv",
        candidate_csv,
        "--out-dir",
        out_dir,
        "--workers",
        str(workers),
        "--timeout",
        str(timeout),
        "--flush-every",
        "10",
    ]
    if resume:
        args.append("--resume")
    return run_script("draft_reference_verifier.py", args, cwd=project_dir, timeout=max(300, timeout * 20))


@mcp.tool()
def init_literature_pool(project_dir: str, pool_dir: str, force: bool = False) -> dict[str, Any]:
    """Initialize a governed literature pool."""
    args = ["init", "--pool-dir", pool_dir]
    if force:
        args.append("--force")
    return run_script("literature_pool.py", args, cwd=project_dir)


@mcp.tool()
def import_verified_to_pool(project_dir: str, pool_dir: str, verified_csv: str, origin: str = "mcp_verified") -> dict[str, Any]:
    """Import verified candidate rows into a governed literature pool."""
    return run_script("literature_pool.py", ["import-csv", "--pool-dir", pool_dir, "--csv", verified_csv, "--origin", origin], cwd=project_dir)


@mcp.tool()
def merge_verified_literature_lanes(
    project_dir: str,
    csv_paths: list[str],
    out_dir: str,
    origin: str = "mcp_verified_union",
    pool_dir: str = "",
    import_to_pool: bool = False,
    skip_missing: bool = True,
) -> dict[str, Any]:
    """Merge accepted verified papers from multiple lanes into one deduped union."""
    args = ["merge-verified", "--out-dir", out_dir, "--origin", origin]
    for csv_path in csv_paths:
        args.extend(["--csv", csv_path])
    if pool_dir:
        args.extend(["--pool-dir", pool_dir])
    if import_to_pool:
        args.append("--import-to-pool")
    if skip_missing:
        args.append("--skip-missing")
    return run_script("literature_pool.py", args, cwd=project_dir)


@mcp.tool()
def export_official_citations(
    project_dir: str,
    verified_csv: str,
    out_dir: str,
    style: str = "vancouver",
    email: str = "",
    timeout: int = 30,
    offline: bool = False,
    allow_derived: bool = False,
    dedupe_by: str = "doi,pmid,title",
) -> dict[str, Any]:
    """Fetch official citation-manager exports for accepted verified papers."""
    args = [
        "export",
        "--verified-csv",
        verified_csv,
        "--out-dir",
        out_dir,
        "--style",
        style,
        "--timeout",
        str(timeout),
        "--dedupe-by",
        dedupe_by,
    ]
    if email:
        args.extend(["--email", email])
    if offline:
        args.append("--offline")
    if allow_derived:
        args.append("--allow-derived")
    return run_script("official_citation_exporter.py", args, cwd=project_dir, timeout=max(300, timeout * 20))


@mcp.tool()
def final_citation_gate(project_dir: str, pool_dir: str, out: str, status: str = "candidate", apply: bool = False) -> dict[str, Any]:
    """Run the published-only final citation gate."""
    args = ["final-gate", "--pool-dir", pool_dir, "--status", status, "--out", out]
    if apply:
        args.append("--apply")
    return run_script("literature_pool.py", args, cwd=project_dir)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
