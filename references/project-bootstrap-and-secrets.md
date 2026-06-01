# Project Bootstrap And Secrets

Use this reference when a review project will run many Python tools, keep long-lived literature state, or call external specialist agents.

## Quick Navigation

- Recommendation: use a lightweight review-root Python project for long-lived state.
- Bootstrap command and consent gate: check before creating `.venv`, `pyproject.toml`, or config files.
- Package boundary: keep skill scripts bundled; keep project state in the review root.
- API key detection: ask only for review-agent keys when not already configured.
- Startup policy: avoid generic venv/API-key behavior outside this review workflow.

## Recommendation

Yes: initialize the review root as a lightweight Python project.

The skill's bundled scripts remain reusable and live inside the skill. The review root should hold project state, a virtual environment, optional project-specific Python package code, literature pools, RAG indexes, agent outputs, cover-letter drafts, and audit logs.

Recommended layout:

```text
review-root/
  .venv/
  pyproject.toml
  .env.example
  .env.local              # optional, local-only secrets; do not commit
  .env.local.example      # template for local secrets
  review_project_config.json
  review_path_aliases.json
  src/review_project/
  review-data/
    00_project/
    01_inputs/
      drafts_raw/
      user_fulltext/
    02_literature/
      draft_assets/
        cleaned_drafts/
      pool/
        pool.json
        recall_runs/
        evidence/
        cards/
        contexts/
        downloads/
        indexes/
        logs/
      supplemental_pool/
        pool.json
        recall_runs/
        cards/
        contexts/
        logs/
    03_framework/
      draft_memory/
      logic_framework/
      display_items/
    04_rag/
    05_audit/
    06_agent_memory/
      shared/
      agents/
  review-work/
    recall_runs/
    agent_orchestration/
    logs/
  review-output/
    manuscript/
    figures/
    tables/
    cover_letter/
    submission/
  review-archive/
```

`文献池` is a legacy/display alias. New projects should use `review-data/02_literature/pool` as the canonical literature-pool path. See `references/data-governance-folder-schema.md`.

## Bootstrap Command

First inspect the project:

```bash
python $SKILL_DIR/scripts/path_schema.py check --project-dir .
python $SKILL_DIR/scripts/project_bootstrap.py check-env --project-dir .
```

Both checks are safe probes. They do not create files, install packages, or print secret values.

## Consent Gate

If `check-env` or `path_schema.py check` reports missing `.venv`, `pyproject.toml`, `review_project_config.json`, `src/review_project`, `review-data`, `review-work`, `review-output`, `review-data/02_literature/pool`, `review-data/03_framework/display_items`, or `review-data/06_agent_memory`, ask the user before running `init`.

Proceed without a separate question only when the user explicitly asked in the current turn to create, initialize, bootstrap, or set up the review project/environment.

Suggested user-facing prompt:

```text
我检测到当前综述根目录还没有 .venv / pyproject.toml / 标准 review-data、review-work、review-output 与规范文献池结构。是否要我现在创建？这会新增项目目录和配置文件，但不会安装联网依赖，也不会写入任何 API key。
```

Initialize the project without installing internet dependencies:

```bash
python $SKILL_DIR/scripts/path_schema.py init --project-dir .
python $SKILL_DIR/scripts/project_bootstrap.py init --project-dir . --package-name review_project --create-venv
python $SKILL_DIR/scripts/agent_memory.py init --project-dir . --agents all
```

Dry run before changing files:

```bash
python $SKILL_DIR/scripts/project_bootstrap.py init --project-dir . --dry-run
```

If optional dependencies are needed, install them only after user approval:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[review]"
```

The generated `pyproject.toml` keeps required dependencies empty because current skill scripts are stdlib-first. Optional `review` dependencies support richer DOCX/PDF parsing, MCP serving, OpenAI-compatible clients, and data-table workflows.

## Package Boundary

Use `src/review_project/` for project-specific code, not for skill code.

Good uses:

- project path constants
- journal-specific configuration
- local data cleaning helpers
- manuscript-specific table/figure builders
- wrappers around generated evidence matrices

Avoid:

- copying all skill scripts into the project
- storing API keys
- putting manuscript prose in Python package files
- mutating user documents without an explicit output path

## External API Key Detection

Default review work uses Codex-subtask packet mode and does not require an API key. Before running optional external specialist agents, inspect configuration:

```bash
python $SKILL_DIR/scripts/review_agent_orchestrator.py verify-config --project-dir .
python $SKILL_DIR/scripts/project_bootstrap.py check-env --project-dir .
```

Recognized optional external variables:

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`, external-mode fallback `https://api.deepseek.com`
- `DEEPSEEK_MODEL`, external-mode fallback `deepseek-v4-pro`
- `REVIEW_AGENT_API_KEY`
- `REVIEW_AGENT_BASE_URL`, external-mode fallback `https://api.deepseek.com`
- `REVIEW_AGENT_MODEL`, external-mode fallback `deepseek-v4-pro`

Preferred locations, in order:

1. OS/process environment variables or a secure MCP secret config.
2. Project-local `.env.local` or `.env`, loaded by `review_agent_orchestrator.py` and `project_bootstrap.py` without printing the key.
3. Explicit `--env-file` for a one-off test.

Keep `.env.example` as a placeholder template only. If a real key is found there, move it to `.env.local` or the OS environment and replace the example value with a placeholder.

If no external key is configured, stay in Codex-subtask packet mode. If an agent API key is already configured, do not ask the user for a key unless the user explicitly requests external acceleration.

If no key is configured:

1. Continue in packet/Codex-subtask mode by default.
2. Tell the user which environment variable is missing.
3. Ask the user to set it in their shell, system environment, `.env.local`, local secrets manager, or future MCP secure config.
4. Do not print, save, or commit real API keys.

Suggested user-facing prompt:

```text
我没有检测到 DEEPSEEK_API_KEY 或 REVIEW_AGENT_API_KEY。默认不需要外部 key：我会先用 packet/Codex 子任务模式生成并审阅任务包。只有当你明确要求外部模型加速且允许数据外发时，才需要在本机环境变量中设置 REVIEW_AGENT_API_KEY 或 DEEPSEEK_API_KEY，然后让我重新检查配置。
```

Avoid asking users to paste API keys directly into normal chat unless the platform provides an explicit secret-entry mechanism. If the user does paste a key, do not echo it; use it only for the current process if needed and do not write it to project files.

## Startup Policy

When the skill is used for planning, prose-only revision, or local citation audit, do not block on API keys.

When the skill is used for `orchestrate`, `run agents`, `DeepSeek`, `MCP`, or any external LLM call:

1. Run `check-env` or `verify-config`.
2. If a key exists, continue.
3. If no key exists, continue with packet/Codex-subtask mode; tell the user how to set a key only when they explicitly request external LLM acceleration.
4. Only request installation or network approval when the specific task needs it.
