---
name: ort-root-workflow
description: Use when modifying anything under /data/qc/dlrm/ORT. Before changing files there, first read ORT/AGENTS.md and ORT/AGENT_WORKLOG.md to understand the repository-wide snapshot, conventions, and recent edits. After changes, append a dated worklog entry and create a git commit containing only the relevant parent-repo files.
---

# ORT Root Workflow

This skill is the workflow guardrail for `/data/qc/dlrm/ORT`.

## Required first step

Before editing anything in this repository:

1. Read `/data/qc/dlrm/ORT/AGENTS.md`
2. Read `/data/qc/dlrm/ORT/AGENT_WORKLOG.md`

Do not skip the worklog. It contains the current repository snapshot and the running modification history.

## While working

- Preserve the parent ORT repository as a self-contained workspace.
- If the task touches a nested repository, also follow that repository's own AGENTS/worklog workflow.
- If you change repository-wide conventions, document the new rule in the worklog entry.

## Before finishing

1. Append a new dated entry to `/data/qc/dlrm/ORT/AGENT_WORKLOG.md`
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
