# Skill Maintenance And Forward Test

Use this reference when the user's goal is to improve `top-journal-review-writer` itself rather than to write a specific manuscript.

## Quick Navigation

- Maintenance mode: edit the skill package, not the active manuscript project unless testing is requested.
- Self-audit: run `skill_audit.py`, `quick_validate.py`, and focused script checks.
- Smoke tests: use minimal project fixtures or explicitly approved real-project tests.
- Forward-test checklist: verify path resolution, triggers, literature flow, full text, RAG, and display items.
- Version hygiene: keep `SKILL.md` light and move detailed commands into references.

## Maintenance Mode

When the user says the work is about improving the skill:

1. Edit files under the skill directory, not the active review project, unless the user explicitly asks for a project smoke test.
2. Prefer adding deterministic scripts or concise references over expanding `SKILL.md`.
3. Keep `SKILL.md` as the routing and operating layer; move detailed procedures to `references/`.
4. Preserve the main contracts: published-only final citation gate, canonical folders, draft-first architecture, full-text handoff, TreeRAG, display items, agent memory, Codex-subtask/packet-first default, optional external LLM acceleration only with approval, and Codex as chief editor.
5. Validate after every substantial change.

## Self-Audit

Run the skill audit to stdout by default:

```bash
python $SKILL_DIR/scripts/skill_audit.py \
  --skill-dir "$SKILL_DIR"
```

If a saved audit artifact is useful, write it to scratch space unless the user explicitly asks for a project-local log:

```bash
python $SKILL_DIR/scripts/skill_audit.py \
  --skill-dir "$SKILL_DIR" \
  --markdown-out C:/tmp/top_journal_review_writer_skill_audit.md
```

Then run the system validator if it exists in the current Codex installation:

```powershell
$env:PYTHONUTF8="1"
python "<skill-creator-dir>/scripts/quick_validate.py" "$SKILL_DIR"
```

Use `PYTHONUTF8=1` on Windows if the validator otherwise reads UTF-8 files with a non-UTF default encoding.

## Smoke Tests

Script syntax:

```bash
python - <<'PY'
from pathlib import Path
root = Path(r"<skill-dir>/scripts")
for path in sorted(root.glob("*.py")):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("OK", path.name)
PY
```

Project-free memory packet:

```bash
python $SKILL_DIR/scripts/agent_memory.py init \
  --project-dir . \
  --memory-dir C:/tmp/top_journal_review_writer_skill_smoke/agent_memory \
  --agents outline_architect,figure_table_designer

python $SKILL_DIR/scripts/review_agent_orchestrator.py plan \
  --project-dir . \
  --out-dir C:/tmp/top_journal_review_writer_skill_smoke/agent_orchestration \
  --agent-memory-dir C:/tmp/top_journal_review_writer_skill_smoke/agent_memory \
  --no-default-inputs \
  --agents outline_architect,figure_table_designer \
  --goal "Smoke-test dynamic agent memory loading."
```

No-key external-agent check:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py verify-config
```

Expected without secrets: Codex-subtask packet mode remains available and `has_api_key=false`. The displayed base URL/model are only external-mode fallbacks for an explicitly approved accelerator.

## Forward-Test Checklist

Use an independent fresh task when possible:

- one prompt asks for draft ingestion and framework discussion from multiple AI drafts
- one prompt asks for literature recall when `paper-search` is unavailable
- one prompt asks for full-text handoff and MinerU parsing
- one prompt asks for multi-agent orchestration with no API key
- one prompt asks for display-item planning with missing RAG/full text and checks `[EVIDENCE GAP]`
- one prompt asks for cover letter and referee suggestions
- one prompt asks for final citation traceback with preprint/unverifiable records

Assess whether Codex:

- loads only the needed reference file
- asks before creating project files when the user did not request initialization
- uses canonical folders
- keeps API keys out of files and logs
- demotes webpages/blogs/preprints correctly
- refuses to cite unverifiable papers
- stops for user alignment before writing a full manuscript from drafts
- writes memory updates as curated state, not as evidence

## Version Hygiene

After adding scripts or references:

- Add the resource to `SKILL.md` only if Codex must discover it.
- Update `agents/openai.yaml` when the user-facing skill promise changes.
- Run `skill_audit.py` and `quick_validate.py`.
- Avoid adding README-style extra files; keep maintenance docs in `references/`.
