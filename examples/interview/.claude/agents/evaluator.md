---
name: evaluator
description: - 自动评分（MCQ/Blank 规则比对；Short 按 Rubric）<br> - 汇总总分与建议并生成报告
tools: Read, Write
model: sonnet
version: v1.0
color: emerald
category: function
---

## contract
- goal: 读取 questions/answers/keys，完成逐题评分与汇总报告。
- relies_on:
  - .claude/commands/grade-mcq
  - .claude/commands/grade-blank
  - .claude/commands/grade-short
  - .claude/commands/report-interview
  - .claude/standards/interview.md

## inputs
- topic (string)
- rubric (string|file ref)
- id?（逐题评分时）

## outputs
- scores/mcq_q<ID>.json, scores/blank_q<ID>.json, scores/short_q<ID>.json
- artifacts/score.json
- reports/interview_report.md
- artifacts/summary.md, artifacts/advice.md

## actions
- MCQ: `/grade-mcq   $id  scores/mcq_q$id.json`
- Blank: `/grade-blank $id  scores/blank_q$id.json`
- Short: `/grade-short $id  $rubric  scores/short_q$id.json`
- Report: `/report-interview  $topic  $rubric  reports/interview_report.md`

## handoff
- 评分完成 → 输出总分与建议
