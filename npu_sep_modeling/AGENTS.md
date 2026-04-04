# Agent Instructions

Scope: everything under `/data/qc/dlrm/ORT/npu_sep_modeling`.

## Required Workflow

Before making any change in this directory:

1. Read [../AGENTS.md](/data/qc/dlrm/ORT/AGENTS.md).
2. Read [../AGENT_WORKLOG.md](/data/qc/dlrm/ORT/AGENT_WORKLOG.md).
3. Read [AGENT_WORKLOG.md](/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md).

Treat the worklogs as the source of truth for repository context and recent edits.

After making any change in this directory:

1. Append a new dated entry to [AGENT_WORKLOG.md](/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md).
2. Include request summary, files changed, behavior changes, validation run, and open risks.
3. Save the change with git in the parent repository rooted at `/data/qc/dlrm/ORT`.

## Git Rules

- Stage only files relevant to the task.
- Do not pull unrelated parent-worktree noise into the commit.
- Do not commit nested repository changes from `single_op_stage1_mlp` or `static_pipeline_eval`.
- Do not amend an earlier commit unless explicitly requested.

## Notes

- If the task changes project conventions, document the new rule in the worklog snapshot or latest entry.
- Keep the NPU truth source limited to `ORT/features_extensible_case_10_4_4_cann` and `ORT/sweep_runs_extensible_case_10_4_4_cann/onnx_profiles`.
