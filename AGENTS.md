# AGENTS.md - Klea Project

This file contains common guidelines for all coding agents working on the project.

## Project Structure

The project contains multiple Python packages:

- [code_pkg/AGENTS.md](code_pkg/AGENTS.md) - AI assisted coding/workflow system
- [mcp_pkg/AGENTS.md](mcp_pkg/AGENTS.md) - MCP server for NeuroML/OSB
- [rag_pkg/AGENTS.md](rag_pkg/AGENTS.md) - Generic RAG implementation
- [utils_pkg/AGENTS.md](utils_pkg/AGENTS.md) - Shared utilities

Consult the relevant package's AGENTS.md for package-specific commands and architecture guidelines.

## Development Workflow

### Session Logging

At the start of each session, check the `.agents/` folder for previous session logs (named `YYYY-MM-DD.md`) to understand where work left off. Read `.agents/README.md` for the session log format and follow it when writing logs — do not write logs from memory.

### Tooling
- **uv** is used as the package manager (not pip directly). Use `uv pip install`, `uv run`, etc.
- **ruff** is used for linting and formatting
- **ty** is used for type checking

### Pre-commit Requirements
- All code must pass ruff linting and formatting
- All code must pass ty type checking
- Import sorting is mandatory
- No trailing whitespace or large files
- Line endings must be Unix format

### Git Conventions
- Never stage or commit without explicit user approval
- When creating new files, use `git add --intent-to-add <file>` so they appear in `git diff`
- Flag large changes and suggest breaking them into smaller, focused commits for provenance and clarity
- Use conventional commit messages when possible
- Include relevant issue numbers in commit messages
- Keep commits focused and atomic
- Ensure all tests pass before pushing

## Security Considerations

### Code Safety
- Avoid eval() and exec() with user input
- Sanitize all file paths and inputs
- Implement proper access controls
- Use HTTPS for all external communications
- Never log sensitive information or credentials

### Comments
- Never remove TODO, FIXME, NOTE, or other comments from the codebase
- Preserve all existing comments during refactoring unless explicitly asked to remove them

This file should be updated as the project evolves. All agents should familiarize themselves with these guidelines before making changes to the codebase.
