# Claude Code CLI 机制全览（中文文档 · 2025-08-18）

> 面向实际使用的“硬核说明书”：讲清楚上下文、记忆、子 Agent、权限/模式、命令与工作流；配足示例与可落地做法。

---

## 1) 总览：Claude Code CLI 是什么

Claude Code 是在你本地终端运行的“可执行 Agent”，能读写项目文件、运行命令、连接 MCP 工具与外部系统，并提供交互式 REPL 与无头（脚本化）两种用法。([Anthropic][1])

---

## 2) 上下文与记忆（Memory）

### 2.1 记忆层级与加载时机

Claude Code 会在**启动时自动加载**多级 `CLAUDE.md` 记忆文件（企业→项目→用户→本地项目），高优先级先加载，后续层级在其上叠加：

| 层级      | 文件位置                                                                                                                                          | 用途 / 范围             |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| 企业策略    | macOS: `/Library/Application Support/ClaudeCode/CLAUDE.md`；Linux: `/etc/claude-code/CLAUDE.md`；Windows: `C:\ProgramData\ClaudeCode\CLAUDE.md` | 组织统一规范              |
| 项目记忆    | `./CLAUDE.md`                                                                                                                                 | 团队共享的项目规范/约定        |
| 用户记忆    | `~/.claude/CLAUDE.md`                                                                                                                         | 个人在所有项目的偏好          |
| （旧）本地项目 | `./CLAUDE.local.md`（已不建议）                                                                                                                     | 个人项目偏好（建议改用 import） |

所有这些文件会被自动读入会话上下文；企业/项目/用户的**优先级**与位置如上。([Anthropic][2])

### 2.2 记忆的 Import 与层级发现

* 在 `CLAUDE.md` 中可用 **`@` 引用**其它文件（含绝对/相对路径、`~` 家目录），并支持**最多 5 层递归 import**。代码/行内 code span 中的 `@` 不会被当作 import。可用 `/memory` 查看已加载的记忆。([Anthropic][2])
* 记忆文件的**发现规则**：从当前工作目录向上递归查找直至根目录（不含根），以及**按需**加载工作目录子树中的 `CLAUDE.md`（当 Claude 读取该子树文件时再加载）。([Anthropic][2])

**快速写入记忆**：输入以 `#` 开头的一行，CLI 会提示把这条规则写入哪个记忆文件；或用 `/init` 生成项目级 `CLAUDE.md` 基础模板。([Anthropic][2])

### 2.3 会话上下文都包含什么

* 记忆文件（见上）
* 你在对话中**显式引用**的文件/资源（见下文 `@ 文件/资源`）
* 工具运行产生的材料（如 `grep`、`git diff`、MCP 资源读取等）
* 历史对话与（可选）**对话压缩**的摘要（`/compact`）
* **注意**：Extended Thinking（扩展思考）会在需要时产生“思考块”（可开启/调整），本质是“在生成前留出更多推理预算”，并非长期记忆；是**当前决策过程**的一部分。([Anthropic][3])

---

## 3) 子 Agent（Subagents）

### 3.1 特性与价值

* 每个子 Agent 有**独立上下文窗口**、**自定义系统提示**与**独立工具权限**；用于把复杂任务委派给专长 Agent，避免污染主线程上下文。([Anthropic][4])

### 3.2 存储位置与优先级

| 类型        | 目录                  | 作用域  | 冲突时优先级 |
| --------- | ------------------- | ---- | ------ |
| 项目子 Agent | `.claude/agents/`   | 当前项目 | 高      |
| 用户子 Agent | `~/.claude/agents/` | 所有项目 | 低      |

子 Agent 文件为 **Markdown + YAML frontmatter**，示例：

```md
---
name: code-reviewer
description: Proactively review code for quality/security; MUST BE USED after edits.
tools: Read, Grep, Glob, Bash  # 省略则继承主线程工具
---

你是资深代码评审……
```

创建/管理推荐使用 `/agents` 交互界面，也可直接增删改文件。([Anthropic][4])

### 3.3 调用与编排

* **自动委派**：Claude 会依据任务描述与子 Agent 描述自动调用。
* **显式调用**：例如“Use the **code-reviewer** subagent …”。
* **链式调用**：可先用 A 子 Agent 分析，再让 B 子 Agent 优化（官方称 *Chaining subagents*）。([Anthropic][4])

---

## 4) 权限、模式与安全

### 4.1 权限系统与模式（Mode）

* 工具权限分层：**只读**（无需授权）/ **Bash**（按命令授权）/ **编辑文件**（会话内授权）。可用 `/permissions` 查看/管理允许（allow）/询问（ask）/拒绝（deny）规则；deny > ask > allow。([Anthropic][5])
* **模式**（`settings.json` 的 `defaultMode` 或 CLI `--permission-mode`）：

  | 模式                                                                                 | 说明                  |
  | ---------------------------------------------------------------------------------- | ------------------- |
  | `default`                                                                          | 首次使用某工具时询问          |
  | `acceptEdits`                                                                      | 自动接受文件编辑许可          |
  | `plan`                                                                             | **只读规划**（不改文件不执行命令） |
  | `bypassPermissions`                                                                | 跳过所有权限询问（高风险）       |
  | 可在交互中通过 **Shift+Tab** 切换（常见：普通 ↔ 自动编辑 ↔ 规划）。([Anthropic][5], [smartscope.blog][6]) |                     |

**设置层级与生效顺序**（高→低）：企业策略 > 命令行参数 > `.claude/settings.local.json` > `.claude/settings.json` > `~/.claude/settings.json`。([Anthropic][7])

**示例：限制与允许**

```json
{
  "permissions": {
    "allow": [
      "Bash(npm run test:*)",
      "Read(~/.zshrc)"
    ],
    "deny": [
      "Bash(curl:*)",
      "Read(./.env)",
      "Read(./secrets/**)"
    ]
  },
  "defaultMode": "plan"
}
```

([Anthropic][7])

### 4.2 追加目录与工作目录

默认仅读写启动目录，可用 `--add-dir` 或 `/add-dir` 扩展访问范围，或在设置中配置 `additionalDirectories`。([Anthropic][5])

---

## 5) 命令体系：内置 Slash、**自定义指令** 与 **输出风格**

### 5.1 常用内置 Slash

| 命令                          | 作用               |
| --------------------------- | ---------------- |
| `/clear`                    | 清空对话历史           |
| `/compact [instructions]`   | 压缩会话并保留要点        |
| `/init`                     | 生成项目 `CLAUDE.md` |
| `/memory`                   | 打开记忆文件编辑         |
| `/agents`                   | 管理子 Agent        |
| `/mcp`                      | 管理 MCP 连接与 OAuth |
| `/permissions`              | 查看/更改权限规则        |
| （完整列表见官方文档）([Anthropic][8]) |                  |

### 5.2 自定义 Slash 命令（Markdown 文件即命令）

* 位置：项目级 `.claude/commands/`；用户级 `~/.claude/commands/`。
* 语法：`/<command> [arguments]`；在命令 Markdown 中可用 `$ARGUMENTS` 占位。
* **`!` 执行 Bash 并把输出纳入上下文**（需在 frontmatter 中声明 `allowed-tools: Bash(...)`）；
* **`@` 引用文件**，把文件内容并入上下文；**也可引用 MCP 资源**（见 §6）。([Anthropic][8])

**示例：Git 提交命令**

```md
---
allowed-tools: Bash(git add:*), Bash(git status:*), Bash(git commit:*)
description: Create a git commit
---

## Context
- Status: !`git status`
- Diff: !`git diff HEAD`
- Branch: !`git branch --show-current`

## Task
基于上述改动撰写一条清晰的提交说明并执行提交。
```

([Anthropic][8])

### 5.3 输出风格（Output Styles）

输出风格是**系统提示级别**的可切换模板，用于“把 Claude Code 变身成不同类型的 Agent”（教学/解释等），与 `CLAUDE.md`（作为用户消息）以及 `--append-system-prompt`（仅 `-p` 打印模式可用）**不同**。可用 `/output-style` 切换或 `/output-style:new` 创建。([Anthropic][9])

---

## 6) 工具与数据：MCP 集成与 `@` 资源

Claude Code 通过 **MCP（Model Context Protocol）** 连接外部工具/数据源（GitHub、Jira、Notion、Stripe、数据库等）。

* 用 `claude mcp add` 添加本地/HTTP/SSE 服务器；支持作用域（local/project/user）与 `.mcp.json`；用 `/mcp` 查看状态与 OAuth。
* **引用资源**：在输入里使用 `@server:protocol://path` 把 MCP 资源并入上下文（自动作为附件读取）。
* MCP 服务器暴露的 Prompt 会以 **`/mcp__server__prompt`** 形式作为 Slash 命令出现。([Anthropic][10])

---

## 7) 工作流（Workflow）：规划、执行、可编排性

### 7.1 规划/执行模式与“扩展思考”

* 通过 **规划模式（Plan Mode）** 在只读下制定方案，再切回普通/自动编辑模式执行；可在 CLI 中用 `--permission-mode plan`，或交互式 **Shift+Tab** 切换。([Anthropic][11], [Anthropic][12])
* 可在 Prompt 中使用“think / think hard / ultrathink”一类关键词提升**扩展思考**预算，适用于复杂问题的更彻底评估。([Anthropic][12])

### 7.2 无头与自动化

* **无头模式**：`claude -p "任务" --max-turns 3`；支持 `--output-format json` 便于脚本解析；可结合 CI（GitHub Actions 等）运行。([Anthropic][13])
* **Hooks**：在生命周期节点（PreToolUse / PostToolUse / Notification / Stop / SessionStart / PreCompact 等）执行**确定性脚本**，可做格式化、阻止敏感编辑、记录日志、定制权限判定等。([Anthropic][14])

**示例：保存每次 Bash 调用**

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          { "type": "command",
            "command": "jq -r '\"\\(.tool_input.command)\"' >> ~/.claude/bash-command-log.txt" }
        ]
      }
    ]
  }
}
```

([Anthropic][14])

### 7.3 能否“嵌套/回归”的工作流？

* 官方机制提供**链式子 Agent**与**Slash/MCP 命令的组合**来实现**分步/嵌套**流程（先分析再优化等）。没有提供“图形化 DSL 的递归工作流引擎”，但通过**子 Agent 链式+自定义命令+Hooks**可以**模拟嵌套与回退（用 Hook 失败码阻断、写入日志、再触发补救命令）**。([Anthropic][4])

---

## 8) 初始化会读哪些文档 / 引用是否入上下文

* **启动时**：自动加载多级 `CLAUDE.md`（含其 `@` import 的文件，遵循 5 层深度限制），并按层级生效。([Anthropic][2])
* **会话中**：

  * **Slash 命令 Markdown** 里的 `@文件`、``!`bash` `` 均会把对应内容或输出**并入上下文**；
  * **普通输入**里引用文件（或 MCP 资源 `@server:...`）也会被读取并入上下文。([Anthropic][8])

---

## 9) CLI 速查（脚本与调试常用）

| 用法                                   | 说明               |         |
| ------------------------------------ | ---------------- | ------- |
| `claude`                             | 启动交互 REPL        |         |
| `claude -p "…" --output-format json` | 无头打印模式（可解析）      |         |
| \`cat file                           | claude -p "…" \` | 管道传入上下文 |
| `claude --model <name>`              | 切换模型             |         |
| `claude --add-dir ../lib`            | 扩展可访问目录          |         |
| `claude -p --max-turns 3 "…" `       | 限制自动回合数          |         |
| （完整命令/参数见官方 CLI 参考）([Anthropic][13]) |                  |         |

---

## 10) 将“规范”固化：代码规范、文档规范、权限与反馈

* **把规范写进 `CLAUDE.md`**（命名、架构、提交信息模板、评审检查表等），并用 `@` 拆分/复用章节。([Anthropic][2])
* **用 Hooks 自动化执行**：保存后自动 `prettier/gofmt`、禁止编辑 `package-lock.json` 或 `.env`、对不合规范的变更给出标准化反馈。([Anthropic][14])
* **权限白/黑名单**：在 `settings.json` 通过 `allow/deny` 精准控制工具与路径，必要时启用 `plan` 模式做只读评审。([Anthropic][7])
* **输出风格**：用 Output Styles 定义“讲法”（解释/教学/审计风格），与工程规范形成互补。([Anthropic][9])

---

## 11) 实操最佳实践（简）

1. **先规划再执行**：进入 Plan 模式，`think hard` 明确方案，再切回执行。([Anthropic][12])
2. **小步快跑 + `/compact`**：阶段性压缩对话，降低上下文噪声。([Anthropic][8])
3. **权责清晰**：把“审查”“优化”“调试”拆成子 Agent；只授予必要工具。([Anthropic][4])
4. **把口头规则写成 Hook**：把“建议/习惯”变成**确定执行**的脚本。([Anthropic][14])
5. **CI 用 `-p`**：输出 JSON、限制回合，收敛脚本行为。([Anthropic][13])

---

## 12) 常见误区 / 直话直说

* **把一切都塞进 `CLAUDE.md`** → 会话冗杂。改用 `@` 分模块，必要时 `/compact`。([Anthropic][2])
* **只靠提示语约束** → 不可靠。规范应落在 **权限与 Hooks**（硬约束），提示语（软约束）只作补充。([Anthropic][5])
* **无头模式当万能编排器** → CLI 提供回合控制与输出格式，但不是状态机/编排引擎；复杂编排请结合 Hooks / 子 Agent 链式 / CI 编排。([Anthropic][13])

---

## 附：最小可用清单

1. `> /init` 生成 `CLAUDE.md`，写入团队规范与`@docs/*`。([Anthropic][2])
2. `> /permissions` 配置允许/拒绝；默认 `plan`。([Anthropic][5])
3. `> /agents` 生成 `code-reviewer`、`debugger` 两个子 Agent。([Anthropic][4])
4. `.claude/commands/commit.md`：用 ``!`git …` `` 聚合上下文自动提交。([Anthropic][8])
5. 在 `settings.json` 加 **PostToolUse** `prettier` Hook 与敏感文件阻断 Hook。([Anthropic][14])
6. CI 中 `claude -p --max-turns 3 --output-format json "run tests & summarize"`。([Anthropic][13])

[1]: https://docs.anthropic.com/en/docs/claude-code/overview?utm_source=chatgpt.com "Claude Code overview"

[2]: https://docs.anthropic.com/en/docs/claude-code/memory "Manage Claude's memory - Anthropic"
[3]: https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips?utm_source=chatgpt.com "Extended thinking tips"
[4]: https://docs.anthropic.com/en/docs/claude-code/sub-agents "Subagents - Anthropic"
[5]: https://docs.anthropic.com/en/docs/claude-code/iam "Identity and Access Management - Anthropic"
[6]: https://smartscope.blog/en/AI/claude-code-auto-permission-guide/?utm_source=chatgpt.com "Claude Code Auto-Permission Guide - 自動実行ガイド - note"
[7]: https://docs.anthropic.com/en/docs/claude-code/settings?utm_source=chatgpt.com "Claude Code settings"
[8]: https://docs.anthropic.com/en/docs/claude-code/slash-commands "Slash commands - Anthropic"
[9]: https://docs.anthropic.com/en/docs/claude-code/output-styles "Output styles - Anthropic"
[10]: https://docs.anthropic.com/en/docs/claude-code/mcp "Connect Claude Code to tools via MCP - Anthropic"
[11]: https://docs.anthropic.com/en/docs/claude-code/sdk?utm_source=chatgpt.com "Claude Code SDK"
[12]: https://www.anthropic.com/engineering/claude-code-best-practices?utm_source=chatgpt.com "Claude Code: Best practices for agentic coding"
[13]: https://docs.anthropic.com/en/docs/claude-code/cli-reference "CLI reference - Anthropic"
[14]: https://docs.anthropic.com/en/docs/claude-code/hooks-guide "Get started with Claude Code hooks - Anthropic"
