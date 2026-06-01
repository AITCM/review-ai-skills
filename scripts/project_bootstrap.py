#!/usr/bin/env python3
"""Bootstrap a review project root and check local agent configuration.

This script is intentionally stdlib-only. It creates a small Python project
surface for long-running review work and reports secret availability without
printing secret values.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import venv
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_PACKAGE = "review_project"
DEFAULT_REVIEW_DATA = "review-data"
DEFAULT_REVIEW_WORK = "review-work"
DEFAULT_REVIEW_OUTPUT = "review-output"
DEFAULT_REVIEW_ARCHIVE = "review-archive"
DEFAULT_LITERATURE_DRAFT_ASSETS = "review-data/02_literature/draft_assets"
DEFAULT_LITERATURE_POOL = "review-data/02_literature/pool"
DEFAULT_LITERATURE_SUPPLEMENTAL_POOL = "review-data/02_literature/supplemental_pool"
DEFAULT_AGENT_MEMORY = "review-data/06_agent_memory"
DEFAULT_DISPLAY_ITEMS = "review-data/03_framework/display_items"
DEFAULT_LOCAL_ENV_FILES = (".env.local", ".env")


def module_name(raw: str) -> str:
    name = re.sub(r"\W+", "_", raw.strip().lower()).strip("_")
    if not name or not re.match(r"^[a-zA-Z_]\w*$", name):
        raise SystemExit(f"Invalid Python package name: {raw!r}")
    return name


def project_name_from_package(package: str) -> str:
    return package.replace("_", "-")


def venv_python(project_dir: Path) -> Path:
    if os.name == "nt":
        return project_dir / ".venv" / "Scripts" / "python.exe"
    return project_dir / ".venv" / "bin" / "python"


def write_file(path: Path, content: str, force: bool, dry_run: bool, actions: list[str]) -> None:
    if path.exists() and not force:
        actions.append(f"exists: {path}")
        return
    actions.append(("would write: " if dry_run else "wrote: ") + str(path))
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ensure_dir(path: Path, dry_run: bool, actions: list[str]) -> None:
    actions.append(("would mkdir: " if dry_run else "mkdir: ") + str(path))
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def pyproject_content(package: str) -> str:
    project = project_name_from_package(package)
    return textwrap.dedent(
        f"""
        [build-system]
        requires = ["setuptools>=68", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{project}"
        version = "0.1.0"
        description = "Local project package for a top-journal review workflow"
        requires-python = ">=3.10"
        dependencies = []

        [project.optional-dependencies]
        review = [
          "python-docx>=1.1.0",
          "requests>=2.31.0",
          "pydantic>=2.0",
          "openai>=1.0.0",
          "mcp>=1.0.0",
          "pypdf>=4.0.0",
          "pymupdf>=1.24.0",
          "pandas>=2.0.0",
          "bibtexparser>=1.4.0"
        ]

        [tool.setuptools.packages.find]
        where = ["src"]
        """
    ).lstrip()


def env_example_content() -> str:
    return textwrap.dedent(
        f"""
        # Template only. Do not put real API keys in .env.example.
        # The default review workflow uses Codex subtasks/packets and does not require an external LLM key.
        # Copy these names to your shell/profile, a secret manager, or .env.local only for optional external acceleration.

        # Generic aliases used by review_agent_orchestrator.py.
        REVIEW_AGENT_API_KEY=
        REVIEW_AGENT_BASE_URL=
        REVIEW_AGENT_MODEL=

        # Optional DeepSeek-compatible backend example.
        DEEPSEEK_API_KEY=
        DEEPSEEK_BASE_URL={DEFAULT_BASE_URL}
        DEEPSEEK_MODEL={DEFAULT_MODEL}

        # Optional metadata API courtesy fields.
        NCBI_EMAIL=
        CROSSREF_MAILTO=
        """
    ).lstrip()


def env_local_example_content() -> str:
    return textwrap.dedent(
        f"""
        # Local secret file template.
        # Copy this file to .env.local and fill values there. Never commit .env.local.
        # Default Codex-subtask mode does not require any values below.

        REVIEW_AGENT_API_KEY=
        REVIEW_AGENT_BASE_URL=
        REVIEW_AGENT_MODEL=

        # Optional DeepSeek-compatible backend.
        DEEPSEEK_API_KEY=
        DEEPSEEK_BASE_URL={DEFAULT_BASE_URL}
        DEEPSEEK_MODEL={DEFAULT_MODEL}
        """
    ).lstrip()


def parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists() or path.is_dir():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def is_placeholder_secret(value: str) -> bool:
    if not value:
        return True
    lowered = value.strip().lower()
    return any(token in lowered for token in ["your_", "example", "placeholder", "replace", "<", ">"])


def local_env_values(project_dir: Path) -> tuple[dict[str, str], list[str]]:
    merged: dict[str, str] = {}
    loaded: list[str] = []
    for name in DEFAULT_LOCAL_ENV_FILES:
        path = project_dir / name
        parsed = parse_env_file(path)
        if parsed:
            loaded.append(str(path))
            merged.update(parsed)
    return merged, loaded


def env_example_has_realish_secret(project_dir: Path) -> bool:
    parsed = parse_env_file(project_dir / ".env.example")
    for key in ["DEEPSEEK_API_KEY", "REVIEW_AGENT_API_KEY", "OPENAI_API_KEY"]:
        value = parsed.get(key, "")
        if value and not is_placeholder_secret(value):
            return True
    return False


def paths_content() -> str:
    return textwrap.dedent(
        """
        from __future__ import annotations

        from pathlib import Path


        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        REVIEW_WORK = PROJECT_ROOT / "review-work"
        REVIEW_DATA = PROJECT_ROOT / "review-data"
        REVIEW_OUTPUT = PROJECT_ROOT / "review-output"
        REVIEW_ARCHIVE = PROJECT_ROOT / "review-archive"
        LITERATURE_POOL = PROJECT_ROOT / "review-data" / "02_literature" / "pool"
        LITERATURE_DRAFT_ASSETS = PROJECT_ROOT / "review-data" / "02_literature" / "draft_assets"
        LITERATURE_SUPPLEMENTAL_POOL = PROJECT_ROOT / "review-data" / "02_literature" / "supplemental_pool"
        DRAFTS_RAW = PROJECT_ROOT / "review-data" / "01_inputs" / "drafts_raw"
        USER_FULLTEXT = PROJECT_ROOT / "review-data" / "01_inputs" / "user_fulltext"
        DRAFT_MEMORY = PROJECT_ROOT / "review-data" / "03_framework" / "draft_memory"
        LOGIC_FRAMEWORK = PROJECT_ROOT / "review-data" / "03_framework" / "logic_framework"
        DISPLAY_ITEMS = PROJECT_ROOT / "review-data" / "03_framework" / "display_items"
        RAG_ROOT = PROJECT_ROOT / "review-data" / "04_rag"
        AUDIT_ROOT = PROJECT_ROOT / "review-data" / "05_audit"
        AGENT_MEMORY = PROJECT_ROOT / "review-data" / "06_agent_memory"
        OUTPUT_MANUSCRIPT = PROJECT_ROOT / "review-output" / "manuscript"
        """
    ).lstrip()


def config_content(package: str) -> str:
    return json.dumps(
        {
            "package": package,
            "path_schema_version": "0.5",
            "review_data": DEFAULT_REVIEW_DATA,
            "review_work": DEFAULT_REVIEW_WORK,
            "review_output": DEFAULT_REVIEW_OUTPUT,
            "review_archive": DEFAULT_REVIEW_ARCHIVE,
            "literature_draft_assets": DEFAULT_LITERATURE_DRAFT_ASSETS,
            "literature_pool": DEFAULT_LITERATURE_POOL,
            "literature_supplemental_pool": DEFAULT_LITERATURE_SUPPLEMENTAL_POOL,
            "agent_memory": DEFAULT_AGENT_MEMORY,
            "draft_memory": "review-data/03_framework/draft_memory",
            "logic_framework": "review-data/03_framework/logic_framework",
            "display_items": DEFAULT_DISPLAY_ITEMS,
            "citation_policy": "published_only_by_default",
            "agent_execution_mode": "codex_subtask_packet_default",
            "external_agent_base_url_if_enabled": DEFAULT_BASE_URL,
            "external_agent_model_if_enabled": DEFAULT_MODEL,
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def cmd_init(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    package = module_name(args.package_name)
    actions: list[str] = []

    dirs = [
        project_dir / "review-data",
        project_dir / "review-data" / "00_project",
        project_dir / "review-data" / "01_inputs" / "drafts_raw",
        project_dir / "review-data" / "01_inputs" / "user_fulltext",
        project_dir / "review-data" / "02_literature" / "draft_assets",
        project_dir / "review-data" / "02_literature" / "draft_assets" / "cleaned_drafts",
        project_dir / "review-data" / "02_literature" / "pool",
        project_dir / "review-data" / "02_literature" / "pool" / "recall_runs",
        project_dir / "review-data" / "02_literature" / "pool" / "evidence",
        project_dir / "review-data" / "02_literature" / "pool" / "cards",
        project_dir / "review-data" / "02_literature" / "pool" / "contexts",
        project_dir / "review-data" / "02_literature" / "pool" / "downloads",
        project_dir / "review-data" / "02_literature" / "pool" / "indexes",
        project_dir / "review-data" / "02_literature" / "pool" / "logs",
        project_dir / "review-data" / "02_literature" / "supplemental_pool",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "recall_runs",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "evidence",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "cards",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "contexts",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "downloads",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "indexes",
        project_dir / "review-data" / "02_literature" / "supplemental_pool" / "logs",
        project_dir / "review-data" / "03_framework" / "draft_memory",
        project_dir / "review-data" / "03_framework" / "logic_framework",
        project_dir / "review-data" / "03_framework" / "display_items",
        project_dir / "review-data" / "04_rag",
        project_dir / "review-data" / "05_audit",
        project_dir / "review-data" / "06_agent_memory",
        project_dir / "review-data" / "06_agent_memory" / "shared",
        project_dir / "review-data" / "06_agent_memory" / "agents",
        project_dir / "review-work",
        project_dir / "review-work" / "recall_runs",
        project_dir / "review-work" / "agent_orchestration",
        project_dir / "review-work" / "logs",
        project_dir / "review-output",
        project_dir / "review-output" / "manuscript",
        project_dir / "review-output" / "figures",
        project_dir / "review-output" / "tables",
        project_dir / "review-output" / "cover_letter",
        project_dir / "review-output" / "submission",
        project_dir / "review-archive",
        project_dir / "src" / package,
    ]
    for directory in dirs:
        ensure_dir(directory, args.dry_run, actions)

    write_file(project_dir / "pyproject.toml", pyproject_content(package), args.force, args.dry_run, actions)
    write_file(project_dir / ".env.example", env_example_content(), args.force, args.dry_run, actions)
    write_file(project_dir / ".env.local.example", env_local_example_content(), args.force, args.dry_run, actions)
    write_file(project_dir / "review_project_config.json", config_content(package), args.force, args.dry_run, actions)
    write_file(project_dir / "src" / package / "__init__.py", f'"""Local package for this review project."""\n\n__all__ = []\n', args.force, args.dry_run, actions)
    write_file(project_dir / "src" / package / "paths.py", paths_content(), args.force, args.dry_run, actions)

    if args.create_venv:
        if args.dry_run:
            actions.append(f"would create venv: {project_dir / '.venv'}")
        else:
            venv.EnvBuilder(with_pip=True, clear=args.clear_venv).create(project_dir / ".venv")
            actions.append(f"created venv: {project_dir / '.venv'}")

    install_hint = str(venv_python(project_dir)) + ' -m pip install -e ".[review]"'
    result = {
        "project_dir": str(project_dir),
        "package": package,
        "venv_python": str(venv_python(project_dir)),
        "created_venv": bool(args.create_venv and not args.dry_run),
        "dry_run": args.dry_run,
        "install_hint": install_hint,
        "actions": actions,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def secret_status(project_dir: Path | None = None) -> dict[str, Any]:
    local_values: dict[str, str] = {}
    loaded_local_env_files: list[str] = []
    env_example_contains_realish_secret = False
    if project_dir is not None:
        local_values, loaded_local_env_files = local_env_values(project_dir)
        env_example_contains_realish_secret = env_example_has_realish_secret(project_dir)

    generic_key = os.environ.get("REVIEW_AGENT_API_KEY", "") or local_values.get("REVIEW_AGENT_API_KEY", "")
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "") or local_values.get("DEEPSEEK_API_KEY", "")
    base_url = (
        os.environ.get("REVIEW_AGENT_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL")
        or local_values.get("REVIEW_AGENT_BASE_URL")
        or local_values.get("DEEPSEEK_BASE_URL")
        or DEFAULT_BASE_URL
    )
    model = (
        os.environ.get("REVIEW_AGENT_MODEL")
        or os.environ.get("DEEPSEEK_MODEL")
        or local_values.get("REVIEW_AGENT_MODEL")
        or local_values.get("DEEPSEEK_MODEL")
        or DEFAULT_MODEL
    )
    return {
        "has_review_agent_api_key": bool(generic_key),
        "has_deepseek_api_key": bool(deepseek_key),
        "has_any_agent_api_key": bool(generic_key or deepseek_key),
        "base_url": base_url,
        "model": model,
        "loaded_local_env_files": loaded_local_env_files,
        "env_example_contains_realish_secret": env_example_contains_realish_secret,
        "secret_policy": "Do not print, store, or commit API keys. Prefer OS environment variables, secure MCP config, or project-local .env.local. Keep .env.example as a placeholder template.",
    }


def cmd_check_env(args: argparse.Namespace) -> int:
    project_dir = Path(args.project_dir).resolve()
    vp = venv_python(project_dir)
    literature_draft_assets_dir = project_dir / DEFAULT_LITERATURE_DRAFT_ASSETS
    literature_pool_dir = project_dir / DEFAULT_LITERATURE_POOL
    literature_supplemental_pool_dir = project_dir / DEFAULT_LITERATURE_SUPPLEMENTAL_POOL
    legacy_literature_pool_dir = project_dir / "\u6587\u732e\u6c60"
    core_paths = {
        ".venv": project_dir / ".venv",
        ".venv_python": vp,
        "pyproject.toml": project_dir / "pyproject.toml",
        "review_project_config.json": project_dir / "review_project_config.json",
        "review-data": project_dir / DEFAULT_REVIEW_DATA,
        "review-work": project_dir / DEFAULT_REVIEW_WORK,
        "review-output": project_dir / DEFAULT_REVIEW_OUTPUT,
        "literature_draft_assets": literature_draft_assets_dir,
        "literature_pool": literature_pool_dir,
        "literature_supplemental_pool": literature_supplemental_pool_dir,
        "agent_memory": project_dir / DEFAULT_AGENT_MEMORY,
        "display_items": project_dir / DEFAULT_DISPLAY_ITEMS,
        "src/review_project": project_dir / "src" / DEFAULT_PACKAGE,
    }
    missing_bootstrap_items = [name for name, path in core_paths.items() if not path.exists()]
    bootstrap_recommended = bool(missing_bootstrap_items)
    suggested_bootstrap_command = (
        f"{sys.executable} {Path(__file__).resolve()} init --project-dir {project_dir} "
        f"--package-name {DEFAULT_PACKAGE} --create-venv"
    )
    suggested_bootstrap_args = [
        sys.executable,
        str(Path(__file__).resolve()),
        "init",
        "--project-dir",
        str(project_dir),
        "--package-name",
        DEFAULT_PACKAGE,
        "--create-venv",
    ]
    status = {
        "project_dir": str(project_dir),
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "venv_exists": (project_dir / ".venv").exists(),
        "venv_python": str(vp),
        "venv_python_exists": vp.exists(),
        "pyproject_exists": (project_dir / "pyproject.toml").exists(),
        "project_config_exists": (project_dir / "review_project_config.json").exists(),
        "missing_bootstrap_items": missing_bootstrap_items,
        "bootstrap_recommended": bootstrap_recommended,
        "requires_user_confirmation_before_init": bootstrap_recommended,
        "suggested_bootstrap_command_after_confirmation": suggested_bootstrap_command if bootstrap_recommended else "",
        "suggested_bootstrap_args_after_confirmation": suggested_bootstrap_args if bootstrap_recommended else [],
        "suggested_user_prompt": (
            "No review project bootstrap changes are needed."
            if not bootstrap_recommended
            else "I detected missing review-project files/folders. Ask the user before creating .venv, pyproject.toml, src/review_project, review-data, review-work, review-output, the governed literature pool, and agent memory folders."
        ),
        "suggested_user_prompt_zh": (
            "当前综述项目不需要创建新的启动文件。"
            if not bootstrap_recommended
            else "我检测到当前综述根目录缺少项目启动文件或规范目录。是否要我创建 .venv、pyproject.toml、src/review_project、review-data、review-work、review-output 和规范文献池等结构？这不会安装联网依赖，也不会写入任何 API key。"
        ),
        "path_schema_version": "0.5",
        "canonical_literature_draft_assets": str(literature_draft_assets_dir),
        "canonical_literature_pool": str(literature_pool_dir),
        "canonical_literature_supplemental_pool": str(literature_supplemental_pool_dir),
        "canonical_agent_memory": str(project_dir / DEFAULT_AGENT_MEMORY),
        "canonical_display_items": str(project_dir / DEFAULT_DISPLAY_ITEMS),
        "legacy_literature_pool_exists": legacy_literature_pool_dir.exists(),
        "legacy_literature_pool": str(legacy_literature_pool_dir),
        "paper_search_cli": shutil.which("paper-search") or "",
        "agent_api": secret_status(project_dir),
    }
    if args.markdown_out:
        lines = [
            "# Review Project Environment Check",
            "",
            f"- Project dir: {status['project_dir']}",
            f"- Python: {status['python']} ({status['python_version']})",
            f"- `.venv` exists: {status['venv_exists']}",
            f"- venv Python: {status['venv_python']} ({status['venv_python_exists']})",
            f"- pyproject exists: {status['pyproject_exists']}",
            f"- bootstrap recommended: {status['bootstrap_recommended']}",
            f"- missing bootstrap items: {', '.join(status['missing_bootstrap_items']) or '[none]'}",
            f"- path schema version: {status['path_schema_version']}",
            f"- canonical draft citation assets: {status['canonical_literature_draft_assets']}",
            f"- canonical literature pool: {status['canonical_literature_pool']}",
            f"- canonical supplemental pool: {status['canonical_literature_supplemental_pool']}",
            f"- canonical agent memory: {status['canonical_agent_memory']}",
            f"- canonical display items: {status['canonical_display_items']}",
            f"- legacy 文献池 exists: {status['legacy_literature_pool_exists']}",
            f"- paper-search CLI: {status['paper_search_cli'] or '[not found]'}",
            f"- agent API key configured: {status['agent_api']['has_any_agent_api_key']}",
            f"- agent base URL: {status['agent_api']['base_url']}",
            f"- agent model: {status['agent_api']['model']}",
            f"- loaded local env files: {', '.join(status['agent_api']['loaded_local_env_files']) or '[none]'}",
            f"- `.env.example` appears to contain a real key: {status['agent_api']['env_example_contains_realish_secret']}",
        ]
        out = Path(args.markdown_out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        status["markdown_out"] = str(out)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap or inspect a local top-journal review project.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create project directories, pyproject.toml, env templates, and optional .venv.")
    p_init.add_argument("--project-dir", default=".")
    p_init.add_argument("--package-name", default=DEFAULT_PACKAGE)
    p_init.add_argument("--create-venv", action="store_true")
    p_init.add_argument("--clear-venv", action="store_true", help="Clear existing .venv before recreating it.")
    p_init.add_argument("--force", action="store_true", help="Overwrite generated files if they already exist.")
    p_init.add_argument("--dry-run", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_check = sub.add_parser("check-env", help="Check project, virtualenv, paper-search CLI, and API key status.")
    p_check.add_argument("--project-dir", default=".")
    p_check.add_argument("--markdown-out", default="")
    p_check.set_defaults(func=cmd_check_env)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
