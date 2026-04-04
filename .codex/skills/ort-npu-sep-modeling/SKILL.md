---
name: ort-npu-sep-modeling
description: Use when modifying anything under /data/qc/dlrm/ORT/npu_sep_modeling. Before changing files there, first read ORT/AGENTS.md, ORT/AGENT_WORKLOG.md, npu_sep_modeling/AGENTS.md, and npu_sep_modeling/AGENT_WORKLOG.md. After changes, append a dated worklog entry and create a git commit containing only the relevant parent-repo files.
---

# ORT NPU Separation Modeling

This skill is the workflow guardrail for `/data/qc/dlrm/ORT/npu_sep_modeling`.

## Required first step

Before editing anything in this directory:

1. Read `/data/qc/dlrm/ORT/AGENTS.md`
2. Read `/data/qc/dlrm/ORT/AGENT_WORKLOG.md`
3. Read `/data/qc/dlrm/ORT/npu_sep_modeling/AGENTS.md`
4. Read `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`

Do not skip the worklogs. They contain the current repository snapshot, the current project snapshot, and the running modification history.

## While working

- Keep the NPU truth source limited to `ORT/features_extensible_case_10_4_4_cann` and `ORT/sweep_runs_extensible_case_10_4_4_cann/onnx_profiles`.
- Treat `CANNExecutionProvider` `Node` events as the source of truth.
- Keep `MemcpyFromHost` / `MemcpyToHost` in a separate `transfer` lane.
- If the task changes conventions, record the new convention in the project worklog.

## Before finishing

1. Append a new dated entry to `/data/qc/dlrm/ORT/npu_sep_modeling/AGENT_WORKLOG.md`
2. Record:
   - request summary
   - files changed
   - behavior changes
   - validation run
   - open risks
3. Create a git commit in the independent repository rooted at `/data/qc/dlrm/ORT`

## Commit Discipline

- Stage only files relevant to the task.
- Do not pull unrelated worktree changes into the commit.
- Do not commit nested repository changes from `single_op_stage1_mlp` or `static_pipeline_eval` into the parent `ORT` repository.
