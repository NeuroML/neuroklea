# AGENTS.md -- Klea monorepo

**NOTE:** `CLAUDE.md` is a symlink to this file (`CLAUDE.md -> AGENTS.md`).
Editing or writing to `CLAUDE.md` will overwrite `AGENTS.md` -- always edit
`AGENTS.md` directly.

Multi-package Python project (setuptools + `setup.cfg`). Each `*_pkg/` is a
separate installable; `code_pkg` and `rag_pkg` depend on `utils_pkg`.

## Packages at a glance

| Dir | Package name | CLI entry |
|-----|-------------|-----------|
| `utils_pkg/` | `klea_utils` | -- (shared lib) |
| `code_pkg/` | `klea_code` | `klea-code` |
| `rag_pkg/` | `klea_rag` | `klea-rag`, `klea-rag-serve` |
| `mcp_pkg/` | `neuroml_mcp` | `nml-mcp` |

Each has its own `AGENTS.md` with architecture details -- refer to those for
package-specific commands, node layout, and conventions.

## Commands

```bash
# Dev install (editable, all packages)
uv pip install -r requirements-dev.txt

# Run all tests across packages (from repo root)
bash scripts/run_tests.sh        # pytest -v -n auto in each *_pkg/tests

# Single package test
cd mcp_pkg && pytest -v          # uses -n 1 (mcp tools are asyncio)
cd utils_pkg && pytest -v

# Run only tests that do NOT need an LLM
pytest -m "not localonly"

# Lint + format
ruff check . --fix
ruff format .
ruff check . --select I --fix    # import sorting

# Type check
ty

# Pre-commit (CI gate)
pre-commit run --all-files
```

`mcp_pkg` pytest uses `addopts = -n 1` -- do **not** run its tests with `-n auto`.

All packages ignore `F403`, `F405` in ruff.

## CI flow

`.github/workflows/ci.yml` (pushes/PRs to main/development/*test*/**feat*/**fix*):
`uv pip install -r ./requirements.txt` -> `ollama pull qwen3:0.6b bge-m3` ->
`bash scripts/run_tests.sh` -> `ruff check . --exit-zero`

`.github/workflows/ruff.yml`: changed-files lint on PRs.

## Config & env loading

Both `KleaCode` and `RAG` orchestrators load configuration via:
1. An env file (`k=v` format, path from `KLEA_CODE_ENV_FILE` / `KLEA_RAG_ENV_FILE` or default `klea_code.env` / `rag.env`)
2. A JSON config file referenced inside the env file

`ty.toml` adds `extra-paths` for all four packages so type-checking resolves
cross-package imports.

## Testing quirks

- Tests marked `localonly` require an LLM -- skipped in CI.
- `utils_pkg/tests/test_stores.py` reads `VS_TEST_CONFIG` env var (default `vector-stores-tests.json`).
- MCP tests are asyncio + single-process (`-n 1`).

## Names to know

Copyright format: `# Copyright 2026 Ankur Sinha <sanjay DOT ankur AT gmail DOT com>`
(`mcp_pkg` additionally requires `#!/usr/bin/env python3` shebang on every `.py` file.)

MCP tool auto-discovery: any function ending `_tool` is registered.

`BaseLangGraph` lives at `utils_pkg/klea_utils/graph/base.py` -- shared
setup -> MCP client -> vector stores -> compile graph template method.

Vector stores use URI-style paths: `chroma:/path/to/dir`, `qdrant:http://...`,
`pgvector:postgresql://...`.

## Session continuity

`.agents/YYYY-MM-DD.md` logs previous work. `.agents/Readme.md` has the format.
Read at session start; write to on completion.

## Git conventions

- `git add --intent-to-add <new-file>` so new files appear in `git diff`.
- Never stage/commit without explicit user approval.
- Conventional commit messages with issue numbers when applicable.

## File conventions

- Use ASCII-only text. No unicode dashes, arrows, ellipsis, or emoticons.
