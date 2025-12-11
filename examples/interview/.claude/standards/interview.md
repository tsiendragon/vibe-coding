# Interview Workflow Standard
**版本**：v1.0
**适用范围**：基于 WOP v0.3 的面试/测评流程（选择题 / 填空题 / 简答题），覆盖出题、作答记录、评分与报告产物。


## 1. 目录与产物规范（Paths & Artifacts）
- 题面（questions/）
  - 选择题：`questions/mcq_q<ID>.md`
  - 填空题：`questions/blank_q<ID>.md`
  - 简答题：`questions/short_q<ID>.md`
- 答案（answers/）
  - `answers/<type>_q<ID>.md`（type ∈ {mcq, blank, short}）
- 答案要点（keys/）
  - 选择题标准答案：`keys/mcq_q<ID>.txt` （内容：单个大写字母 A/B/C/D）
  - 填空题可接受答案：`keys/blank_q<ID>.txt`（多答案用 `||` 分隔）
- 评分（scores/）
  - `scores/mcq_q<ID>.json`、`scores/blank_q<ID>.json`、`scores/short_q<ID>.json`
- 报告与总分
  - 总报告：`reports/interview_report.md`
  - 聚合分数：`artifacts/score.json`
  - 摘要与建议（可选）：`artifacts/summary.md`、`artifacts/advice.md`

> **统一要求**：UTF-8 编码、Unix 换行。所有文件路径相对仓库根目录。

---

## 2. 题面结构（Question Formats）

### 2.1 选择题（MCQ）
**文件**：`questions/mcq_q<ID>.md`
**MUST**
- 第一行以 `# Q<ID>. <简短题干>` 开头（≤120 字）。
- 包含 **四个**选项，行首以 `A. `、`B. `、`C. `、`D. ` 标记。
- 仅 **一个** 正确答案；对应 `keys/mcq_q<ID>.txt` 中的单字母（A–D）。
- 题干与选项清晰、无歧义、无“以上皆是/皆非”。

**示例**
```md
# Q1. 在小批量梯度下降（Mini-batch SGD）中，批量大小主要影响哪一项？
A. 模型参数数量
B. 每次参数更新的方差
C. 学习率的上限
D. 激活函数的选择
````

`keys/mcq_q1.txt`：

```
B
```

### 2.2 填空题（Blank)

**文件**：`questions/blank_q<ID>.md`
**MUST**

* 用 `____` 表示空位；题干 ≤120 字；避免多空位（如需要，多空位依次编号：`____(1)`、`____(2)`）。
* `keys/blank_q<ID>.txt` 列出**可接受答案集合**，用 `||` 分隔；大小写不敏感，前后空白忽略。

**示例**

```md
在优化算法中，____ 常被用来应对非凸目标的局部极小问题。
```

`keys/blank_q1.txt`：

```
随机重启||random restart
```

### 2.3 简答题（Short)

**文件**：`questions/short_q<ID>.md`
**MUST**

* 第一行 `# Q<ID>. <题目>`；题干聚焦一个主题，避免“过宽泛”。
* 明确回答要求（例如“给出2-3个要点，并说明适用场景”）。

**示例**

```md
# Q1. 简述正则化(L1/L2)在过拟合防治中的作用，并各举一个适用场景。
```

---

## 3. 答案记录（Answer Recording）

**文件**：`answers/<type>_q<ID>.md`
**MUST**

* **选择题**：第一行仅保留一个字母 `A|B|C|D`。
* **填空题**：第一行填写文本答案（大小写不敏感；仅第一行参与自动判分）。
* **简答题**：自由文本，可分点；建议 80–200 字。

---

## 4. 评分输出（Scoring Outputs）

### 4.1 MCQ/Blank JSON 结构

```json
{
  "type": "mcq | blank",
  "id": 1,
  "key": "C",               // mcq
  "keys": ["sgd", "stochastic gradient descent"], // blank
  "answer": "B" | "sgd",
  "correct": 0 | 1,
  "score": 0 | 1
}
```

### 4.2 Short JSON 结构（基于 Rubric）

```json
{
  "type": "short",
  "id": 1,
  "dimensions": {
    "concept_accuracy": 0,  // 0..5
    "clarity": 0,
    "depth_breadth": 0,
    "practical_relevance": 0,
    "logic_examples": 0
  },
  "overall": 0,             // 0..100（推荐=维度均值*20）
  "comments": {
    "strengths": ["...","..."],
    "improvements": ["...","..."]
  }
}
```

### 4.3 汇总分数（artifacts/score.json）

```json
{
  "topic": "machine learning basics",
  "counts": { "mcq": 3, "blank": 2, "short": 2 },
  "totals": { "mcq": 2, "blank": 2, "short_avg": 78 },
  "overall": 82
}
```

---

## 5. 报告（reports/interview\_report.md）

**MUST**

* 标题：`# Interview Report - <topic>`
* 概览：题量、答题通过情况（MCQ/Blank 正确数）、简答题平均分。
* 优势与改进（各 3 条以内）。
* 建议与资源链接（至少 3 条，含文档/练习方向）。
* 可附附录：逐题要点（题面引用 + 作答摘要 + 得分/评语）。

---

## 6. 质量门（Quality Gates）

**MUST**

* 每个题型的 `questions/*` 与 `keys/*`（如适用）文件**存在且格式合法**。
* 选择题仅一项正确，`keys/mcq_q<ID>.txt` 为单字母 A–D。
* 填空题 `keys/blank_q<ID>.txt` 至少一个候选答案，使用 `||` 分隔。
* `answers/` 与 `scores/` 一一对应，评分 JSON 可被 `jq` 正常解析。
* 报告文件存在，且包含总分与建议。

**MUST NOT**

* 题干或选项出现歧义/双关/“以上皆是/皆非”。
* 题目超范围或与主题无关。
* 评分 JSON 缺维度或字段名不一致。

---

## 7. 最小机检脚本（可选，用作 WOP checks）

保存为：`.claude/commands/interview-lint`

```bash
#!/usr/bin/env bash
# Minimal linter for interview workflow
set -euo pipefail

err=0

check_mcq() {
  local id="$1"
  local q="questions/mcq_q${id}.md"
  local k="keys/mcq_q${id}.txt"
  [[ -f "$q" ]] || { echo "MISSING $q"; err=1; return; }
  [[ -f "$k" ]] || { echo "MISSING $k"; err=1; return; }
  head -n1 "$q" | grep -Eq "^# Q${id}\. " || { echo "BAD TITLE $q"; err=1; }
  grep -Eq "^A\. " "$q" && grep -Eq "^B\. " "$q" && grep -Eq "^C\. " "$q" && grep -Eq "^D\. " "$q" || { echo "MCQ OPTIONS BAD in $q"; err=1; }
  tr -d '\r\n ' < "$k" | grep -Eq "^[ABCD]$" || { echo "MCQ KEY BAD in $k"; err=1; }
}

check_blank() {
  local id="$1"
  local q="questions/blank_q${id}.md"
  local k="keys/blank_q${id}.txt"
  [[ -f "$q" ]] || { echo "MISSING $q"; err=1; return; }
  [[ -f "$k" ]] || { echo "MISSING $k"; err=1; return; }
  grep -q "____" "$q" || { echo "NO BLANK placeholder in $q"; err=1; }
  [[ -n "$(tr -d '\r\n ' < "$k")" ]] || { echo "BLANK KEYS EMPTY in $k"; err=1; }
}

check_short() {
  local id="$1"
  local q="questions/short_q${id}.md"
  [[ -f "$q" ]] || { echo "MISSING $q"; err=1; return; }
  head -n1 "$q" | grep -Eq "^# Q${id}\. " || { echo "BAD TITLE $q"; err=1; }
}

# Auto-discover ids by filenames
for f in questions/mcq_q*.md 2>/dev/null; do id="${f##*mcq_q}"; id="${id%.md}"; check_mcq "$id"; done
for f in questions/blank_q*.md 2>/dev/null; do id="${f##*blank_q}"; id="${id%.md}"; check_blank "$id"; done
for f in questions/short_q*.md 2>/dev/null; do id="${f##*short_q}"; id="${id%.md}"; check_short "$id"; done

[[ $err -eq 0 ]] && echo "interview-lint: PASS" || exit 1
```

赋权：

```bash
chmod +x .claude/commands/interview-lint
```

**WOP 集成（示例）**

```yaml
refs:
  standards:
    - id: std.interview.min
      name: "面试流程最小规范"
      uri: ".claude/standards/interview.md"
      enforce:
        when: ["s_mcq_loop","s_blank_loop","s_short_loop","s_report"]
        checks:
          - type: tool
            tool: "@tool.interview-lint"
  tools:
    - id: tool.interview-lint
      kind: shell
      entry: ".claude/commands/interview-lint"
```

---

## 8. 出题与评分命令约束（参考）

* `/ask-mcq  <topic>  <id>  questions/mcq_q<id>.md  keys/mcq_q<id>.txt`
  MUST 生成 4 选项，keys 为单字母。
* `/ask-blank <topic>  <id>  questions/blank_q<id>.md keys/blank_q<id>.txt`
  MUST 使用 `____`，keys 用 `||` 分隔。
* `/ask-short <topic> <id>  questions/short_q<id>.md`
  MUST 规定回答范围与长度建议。
* `/record <type> <id> <answer> answers/<type>_q<id>.md`
  MUST 将 Gate 评论原文写到答案文件**第一行**（MCQ/blank 自动评分依赖）。
* `/grade-mcq`、`/grade-blank`、`/grade-short`
  MUST 产出符合本标准的 JSON；失败时返回非 0。
