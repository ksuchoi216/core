# CORE SUBMODULE GUIDANCE

`core/` is a git submodule (`git@github.com:ksuchoi216/core.git`, branch `develop`). Treat it as shared infrastructure, not project-local product code.

## Scope

- `support/`: config loading, file IO, deterministic seeds.
- `llm/`: model config handling, prompt chains, graph node helpers, text utilities.
- `langfuse/`: prompt management and tracing runners.
- `aws/`: S3 artifact and transfer helpers.
- `vectordb/`: vector database helpers.
- `commands/`: CLI helpers for prompt download/upload/key creation.

## Editing Rules

- Avoid changing `core/` to satisfy a narrow AlgoCoach behavior unless the bug is truly in shared infrastructure.
- If a change is needed here, check callers in both this repo and the submodule shape before editing.
- Preserve strict config validation in `core.support.config`; model YAML keys must match the provider-specific Pydantic schemas.
- Keep Langfuse disabled-path behavior working. Tests and local runs may not have Langfuse credentials.
- Prefer adding project-specific logic in `src/` or `tasks/` over expanding generic helpers here.

## Important Interfaces

- `core.support.config.load_config(is_test=True)` loads `configs/general.yaml` and `configs/models.yaml`.
- `core.langfuse.run_graph_with_langfuse` is the graph execution surface task runners are expected to call.
- `core.support.file.load_file` and `save_file` are the local file IO helpers used by tests and runners.

