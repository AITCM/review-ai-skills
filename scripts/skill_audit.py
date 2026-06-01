#!/usr/bin/env python3
"""Audit the top-journal-review-writer skill package.

This script checks the skill as a reusable package: referenced resources,
Python syntax, key workflow contracts, and UI metadata presence. It does not
touch a review project.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_CONTRACTS = {
    "agent_memory_path": "review-data/06_agent_memory",
    "display_items_path": "review-data/03_framework/display_items",
    "literature_pool_path": "review-data/02_literature/pool",
    "deepseek_default_model": "deepseek-v4-pro",
    "published_only_policy": "published-only",
}

FORBIDDEN_MARKDOWN_PATTERNS = {
    "relative_script_command": r"\bpython\s+scripts[/\\][^\s`]+\.py",
    "relative_script_run_reference": r"\brun\s+`scripts[/\\][^`]+\.py\b",
    "legacy_draft_dir_example": r"--draft-dir\s+\./drift\b",
    "project_local_maintenance_audit": r"--markdown-out\s+\./review-work/",
    "fake_email_placeholder": r"your@email\.com",
}

FORBIDDEN_DESCRIPTION_TRIGGERS = [
    "venv",
    "pyproject",
    "api key",
    "subagent",
]

SKILL_TOPIC_SPECIFIC_EXAMPLES = [
    "self-evolving",
    "TCM",
    "Traditional Chinese Medicine",
    "Nature Reviews Bioengineering",
    "2026",
]

MAX_SKILL_LINES = 220
MAX_SKILL_CHARS = 30000
MAX_COMMAND_INDEX_COMMANDS = 12
MAX_COMMAND_INDEX_LINES = 45
LONG_REFERENCE_LINES = 100


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def line_count(text: str) -> int:
    return len(text.splitlines())


def markdown_section(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    next_heading = re.search(r"^##\s+", text[match.end() :], flags=re.MULTILINE)
    if not next_heading:
        return text[match.end() :]
    return text[match.end() : match.end() + next_heading.start()]


def has_quick_navigation(text: str) -> bool:
    head = "\n".join(text.splitlines()[:60])
    return bool(re.search(r"^##\s+(Quick Navigation|Contents|Table of Contents|快速导航)\s*$", head, flags=re.MULTILINE))


def find_references(text: str) -> set[str]:
    patterns = [
        ("", r"`(references/[^`]+?\.md)`"),
        ("", r"`(scripts/[^`]+?\.py)`"),
        ("", r"python\s+(scripts/[^ \n\r\t]+?\.py)"),
        ("", r"python\s+[^ \n\r\t]*top-journal-review-writer/(scripts/[^ \n\r\t]+?\.py)"),
        ("scripts/", r"<S>/([^`\s]+?\.py)\b"),
        ("scripts/", r"\$SKILL_DIR[/\\]scripts[/\\]([^\"'`\s]+?\.py)\b"),
    ]
    found: set[str] = set()
    for prefix, pattern in patterns:
        for match in re.findall(pattern, text):
            found.add((prefix + match).replace("\\", "/"))
    return found


def compile_python(path: Path) -> str:
    try:
        compile(read_text(path), str(path), "exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive for odd encodings
        return f"read/compile error: {exc}"
    return ""


def frontmatter_ok(skill_md: Path) -> tuple[bool, dict[str, str]]:
    text = read_text(skill_md)
    if not text.startswith("---\n"):
        return False, {}
    end = text.find("\n---", 4)
    if end < 0:
        return False, {}
    meta: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"')
    return bool(meta.get("name") and meta.get("description")), meta


def audit(skill_dir: Path) -> dict[str, Any]:
    skill_dir = skill_dir.resolve()
    skill_md = skill_dir / "SKILL.md"
    references_dir = skill_dir / "references"
    scripts_dir = skill_dir / "scripts"
    agents_yaml = skill_dir / "agents" / "openai.yaml"
    issues: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if not skill_md.exists():
        issues.append({"code": "missing_skill_md", "message": f"Missing {skill_md}"})
        return {"skill_dir": str(skill_dir), "issues": issues, "warnings": warnings}

    skill_text = read_text(skill_md)
    skill_lines = line_count(skill_text)
    if skill_lines > MAX_SKILL_LINES:
        warnings.append(
            {
                "code": "skill_md_too_long",
                "message": f"SKILL.md has {skill_lines} lines; keep the routing layer under {MAX_SKILL_LINES} lines.",
            }
        )
    if len(skill_text) > MAX_SKILL_CHARS:
        warnings.append(
            {
                "code": "skill_md_too_large",
                "message": f"SKILL.md has {len(skill_text)} characters; move detailed procedures to references.",
            }
        )
    command_index = markdown_section(skill_text, "Minimal Command Index")
    if command_index:
        command_count = len(re.findall(r"\bpython\s+[\"']?\$SKILL_DIR[/\\]scripts[/\\][^ \n\r\t\"']+\.py", command_index))
        command_lines = line_count(command_index)
        if command_count > MAX_COMMAND_INDEX_COMMANDS:
            warnings.append(
                {
                    "code": "command_index_too_many_commands",
                    "message": f"Minimal Command Index lists {command_count} script commands; keep at most {MAX_COMMAND_INDEX_COMMANDS}.",
                }
            )
        if command_lines > MAX_COMMAND_INDEX_LINES:
            warnings.append(
                {
                    "code": "command_index_too_long",
                    "message": f"Minimal Command Index has {command_lines} lines; keep detailed command chains in references.",
                }
            )
    ok_frontmatter, meta = frontmatter_ok(skill_md)
    if not ok_frontmatter:
        issues.append({"code": "bad_frontmatter", "message": "SKILL.md frontmatter must contain name and description."})
    else:
        description = meta.get("description", "")
        description_lower = description.lower()
        for trigger in FORBIDDEN_DESCRIPTION_TRIGGERS:
            if re.search(rf"\b{re.escape(trigger)}\b", description_lower):
                issues.append(
                    {
                        "code": "overbroad_trigger",
                        "message": f"Frontmatter description contains generic trigger: {trigger}",
                    }
                )

    if not agents_yaml.exists():
        warnings.append({"code": "missing_openai_yaml", "message": "agents/openai.yaml is missing."})

    all_scripts = sorted(scripts_dir.glob("*.py")) if scripts_dir.exists() else []
    script_results = []
    for script in all_scripts:
        error = compile_python(script)
        script_results.append({"path": str(script), "ok": not error, "error": error})
        if error:
            issues.append({"code": "script_syntax", "message": f"{script.name}: {error}"})

    all_references = sorted(references_dir.glob("*.md")) if references_dir.exists() else []
    if len(all_references) < 5:
        warnings.append({"code": "few_references", "message": "Expected multiple reference files for progressive disclosure."})
    if len(all_scripts) < 5:
        warnings.append({"code": "few_scripts", "message": "Expected bundled scripts for deterministic workflow steps."})
    reference_navigation: list[dict[str, Any]] = []
    for reference in all_references:
        text = read_text(reference)
        lines = line_count(text)
        has_nav = has_quick_navigation(text)
        reference_navigation.append({"path": str(reference), "lines": lines, "has_quick_navigation": has_nav})
        if lines > LONG_REFERENCE_LINES and not has_nav:
            warnings.append(
                {
                    "code": "long_reference_missing_navigation",
                    "message": f"{reference.name} has {lines} lines and no Quick Navigation near the top.",
                }
            )

    reference_text = "\n".join(read_text(path) for path in all_references)
    markdown_text = skill_text + "\n" + reference_text
    referenced = sorted(find_references(markdown_text))
    missing_refs = []
    for rel in referenced:
        path = skill_dir / rel
        if not path.exists():
            missing_refs.append(rel)
    for rel in missing_refs:
        issues.append({"code": "missing_referenced_resource", "message": rel})

    combined = markdown_text + "\n" + "\n".join(read_text(path) for path in all_scripts)
    for code, needle in REQUIRED_CONTRACTS.items():
        if needle not in combined:
            issues.append({"code": f"missing_contract_{code}", "message": f"Could not find required contract text: {needle}"})

    for code, pattern in FORBIDDEN_MARKDOWN_PATTERNS.items():
        if re.search(pattern, markdown_text):
            issues.append({"code": code, "message": f"Forbidden markdown pattern found: {pattern}"})

    for text in SKILL_TOPIC_SPECIFIC_EXAMPLES:
        if text in skill_text:
            warnings.append({"code": "topic_specific_skill_example", "message": f"Topic/date-specific example in SKILL.md: {text}"})

    mode_names = re.findall(r"^- `([^`]+)`:", skill_text, flags=re.MULTILINE)
    for expected in ["skill-maintenance", "draft-memory", "agent-memory", "display-items", "orchestrate", "audit", "cover-letter"]:
        if expected not in mode_names:
            warnings.append({"code": "missing_mode", "message": f"Mode not listed in SKILL.md: {expected}"})

    return {
        "skill_dir": str(skill_dir),
        "metadata": meta,
        "referenced_resources": referenced,
        "script_count": len(all_scripts),
        "reference_count": len(all_references),
        "skill_lines": skill_lines,
        "skill_chars": len(skill_text),
        "command_index_commands": len(re.findall(r"\bpython\s+[\"']?\$SKILL_DIR[/\\]scripts[/\\][^ \n\r\t\"']+\.py", command_index)) if command_index else 0,
        "command_index_lines": line_count(command_index) if command_index else 0,
        "reference_navigation": reference_navigation,
        "scripts": script_results,
        "issues": issues,
        "warnings": warnings,
        "ok": not issues,
    }


def write_markdown(path: Path, data: dict[str, Any]) -> None:
    lines = [
        "# Skill Audit",
        "",
        f"- Skill dir: `{data['skill_dir']}`",
        f"- OK: {data.get('ok', False)}",
        f"- Scripts: {data.get('script_count', 0)}",
        f"- References: {data.get('reference_count', 0)}",
        f"- SKILL.md lines/chars: {data.get('skill_lines', 0)} / {data.get('skill_chars', 0)}",
        f"- Minimal command index commands/lines: {data.get('command_index_commands', 0)} / {data.get('command_index_lines', 0)}",
        f"- Issues: {len(data.get('issues', []))}",
        f"- Warnings: {len(data.get('warnings', []))}",
        "",
        "## Issues",
        "",
    ]
    if data.get("issues"):
        for item in data["issues"]:
            lines.append(f"- `{item['code']}`: {item['message']}")
    else:
        lines.append("- [none]")
    lines.extend(["", "## Warnings", ""])
    if data.get("warnings"):
        for item in data["warnings"]:
            lines.append(f"- `{item['code']}`: {item['message']}")
    else:
        lines.append("- [none]")
    lines.extend(["", "## Referenced Resources", ""])
    for rel in data.get("referenced_resources", []):
        lines.append(f"- `{rel}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the top-journal-review-writer skill package.")
    parser.add_argument("--skill-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--markdown-out", default="")
    parser.add_argument("--fail-on-warning", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = audit(Path(args.skill_dir))
    if args.markdown_out:
        write_markdown(Path(args.markdown_out), data)
    print(json.dumps(data, ensure_ascii=False, indent=2))
    if data["issues"] or (args.fail_on_warning and data["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
