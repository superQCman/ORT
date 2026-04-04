# Agent Instructions

Scope: everything under `/data/qc/dlrm/ORT/single_op_stage1_mlp`.

## Required Workflow

Before making any change in this directory:

1. Read [AGENT_WORKLOG.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md).
2. Read at least the `Project Snapshot` and the latest `Change History` entries.
3. Treat that file as the current source of truth for project context, conventions, and recent decisions.

After making any change in this directory:

1. Append a new dated entry to [AGENT_WORKLOG.md](/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md).
2. Include: request summary, files changed, behavior changes, validation run, and open risks.
3. Save the change with git in the independent repository rooted at `/data/qc/dlrm/ORT/single_op_stage1_mlp` before finishing the task.

## Git Rules

- Commit after each completed modification task affecting this directory.
- Do not commit these changes into the parent `ORT` repository.
- Use the independent git repository rooted at `/data/qc/dlrm/ORT/single_op_stage1_mlp`.
- Stage only the files relevant to the task.
- Do not include unrelated dirty worktree changes from elsewhere in the repo.
- Do not amend an earlier commit unless the user explicitly asks for it.
- If a commit cannot be created, explain why in the final response and record that in the worklog.

## Expected Commit Scope

When the task is limited to this project, the commit should usually include only:

- files under `/data/qc/dlrm/ORT/single_op_stage1_mlp`
- the local skill under `/data/qc/dlrm/ORT/single_op_stage1_mlp/.codex/skills/ort-single-op-stage1-mlp`

## Notes

- The worklog is not optional. Update it every time.
- If the task changes project conventions, document the new convention in the worklog snapshot or latest entry.
