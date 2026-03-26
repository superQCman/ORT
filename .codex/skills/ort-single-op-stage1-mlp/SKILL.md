---
name: ort-single-op-stage1-mlp
description: Use when modifying anything under /data/qc/dlrm/ORT/single_op_stage1_mlp. Before changing files there, first read AGENTS.md and AGENT_WORKLOG.md in that directory to understand the project snapshot, conventions, and recent edits. After changes, append a dated worklog entry and create a git commit in the independent repository rooted at /data/qc/dlrm/ORT/single_op_stage1_mlp.
---

# ORT Single-Op Stage-1 MLP

This skill is the workflow guardrail for `/data/qc/dlrm/ORT/single_op_stage1_mlp`.

## Required first step

Before editing anything in this directory:

1. Read `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENTS.md`
2. Read `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`

Do not skip the worklog. It contains the current project snapshot and the running modification history.

## While working

- Preserve the project as a self-contained pipeline.
- Follow the documented feature, label, and profile-cleaning conventions unless the user explicitly asks to change them.
- If you change conventions, document the new rule in the worklog entry.

## Before finishing

1. Append a new dated entry to `/data/qc/dlrm/ORT/single_op_stage1_mlp/AGENT_WORKLOG.md`
2. Record:
   - request summary
   - files changed
   - behavior changes
   - validation run
   - open risks
3. Create a git commit in the independent repository rooted at `/data/qc/dlrm/ORT/single_op_stage1_mlp`

## Commit discipline

- Stage only files related to the task.
- Do not pull in parent `ORT` repository noise.
- Do not amend older commits unless the user explicitly asks.
