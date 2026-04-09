# Agent Instructions

Scope: everything under `/data/qc/dlrm/ORT`.

## Required Workflow

Before making any change anywhere in this repository:

1. Read [AGENT_WORKLOG.md](/data/qc/dlrm/ORT/AGENT_WORKLOG.md).
2. Read the latest `Project Snapshot` and the latest `Change History` entries.
3. Treat that file as the current source of truth for repository-wide context, conventions, and recent edits.

If the task touches a nested git repository under `ORT/` such as:

- `/data/qc/dlrm/ORT/CPU_Perf_Model`
- `/data/qc/dlrm/ORT/single_op_stage1_mlp`
- `/data/qc/dlrm/ORT/static_pipeline_eval`

then also read that repository's own `AGENTS.md` and `AGENT_WORKLOG.md` before changing files there.

After making any change in this repository:

1. Append a new dated entry to [AGENT_WORKLOG.md](/data/qc/dlrm/ORT/AGENT_WORKLOG.md).
2. Include:
   - request summary
   - files changed
   - behavior changes
   - validation run
   - open risks
3. Save the change with git in the independent repository rooted at `/data/qc/dlrm/ORT` before finishing the task.

## Git Rules

- Stage only files relevant to the task.
- Do not pull unrelated parent-worktree noise into the commit.
- Do not amend an earlier commit unless the user explicitly asks.
- Do not commit nested repository changes from `single_op_stage1_mlp` or `static_pipeline_eval` into the parent `ORT` repository.

## Expected Commit Scope

When the task is limited to this repository, the commit should usually include only:

- files under `/data/qc/dlrm/ORT`
- the local root skill under `/data/qc/dlrm/ORT/.codex/skills/ort-root-workflow`

## Notes

- The nested repositories inside `ORT/` remain independent git repositories with their own worklogs and commit rules.
- Keep the repository self-contained. If a change needs a cross-repo dependency, document it explicitly in the worklog entry.
