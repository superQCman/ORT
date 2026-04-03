---
name: ort-static-pipeline-eval
description: Use when modifying anything under /data/qc/dlrm/ORT/static_pipeline_eval. Before changing files there, first read ORT/static_pipeline_eval/AGENTS.md and ORT/static_pipeline_eval/AGENT_WORKLOG.md to understand the project snapshot, conventions, and recent edits. After changes, append a dated worklog entry and create a git commit containing only the relevant files.
---

# ORT Static Pipeline Eval

This skill is the workflow guardrail for `/data/qc/dlrm/ORT/static_pipeline_eval`.

## Required first step

Before editing anything in this directory:

1. Read `/data/qc/dlrm/ORT/static_pipeline_eval/AGENTS.md`
2. Read `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`

Do not skip the worklog. It contains the current project snapshot and running modification history.

## While working

- Keep the project self-contained.
- Preserve the current `v1` assumptions unless the user explicitly asks to change them.
- If scheduling semantics or report semantics change, document the new rule in the worklog entry.

## Before finishing

1. Append a new dated entry to `/data/qc/dlrm/ORT/static_pipeline_eval/AGENT_WORKLOG.md`
2. Record:
   - request summary
   - files changed
   - behavior changes
   - validation run
   - open risks
3. Create a git commit with only the relevant files

## Commit discipline

- Stage only files related to the task.
- Do not pull unrelated repo noise into the commit.
- Do not amend older commits unless the user explicitly asks.
