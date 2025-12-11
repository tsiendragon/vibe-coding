# Codex CLI 介绍文档

明白了：你指的是 **OpenAI 的 Codex CLI**。在 Ubuntu 远程机器（无浏览器/无桌面）上使用它，核心是**在能开浏览器的本机完成登录**，再把认证文件拷到远端；或用 API Key。下面给出最稳妥做法。

## 一、在远端 Ubuntu 安装 Codex CLI

```bash
# 任选其一
npm install -g @openai/codex
# 或者（推荐生产环境用 release 二进制）
# 到 GitHub Release 下载 codex-x86_64-unknown-linux-musl.tar.gz 解压后放到 /usr/local/bin/codex
```

官方说明：支持 macOS/Linux，首次运行会提示认证。([OpenAI开发者][1])

---

## 二、无头（headless）登录的两种方式

### 方案 A：本地登录 + 拷贝凭证（最常用）

1. 在**本地**（能开浏览器的机器）安装并登录：

```bash
npm i -g @openai/codex
codex   # 选择 "Sign in with ChatGPT" 按流程完成浏览器登录
```

2. 登录后会生成本地凭证文件 `~/.codex/auth.json`。将它**安全复制**到远端同一路径：

```bash
scp ~/.codex/auth.json <user>@<remote>:/home/<user>/.codex/auth.json
ssh <user>@<remote> 'chmod 600 ~/.codex/auth.json'
```

3. 登入远端验证：

```bash
ssh <user>@<remote>
codex "echo hello"   # 或 codex --help 试运行
```

（GitHub 仓库文档提供了 *Login on a "Headless" machine* 与 *Authentication* 专章；该方法即其建议之一。）([GitHub][2])

### 方案 B：使用 OpenAI API Key（备选）

* 将 `OPENAI_API_KEY` 配置在远端环境（例如 `~/.bashrc`）并按文档的 **API-based authentication** 指南运行。
  *注意：Codex 推荐“Sign in with ChatGPT”方式；API Key 需要额外设置且行为可能有差异。* ([OpenAI开发者][1])

---

## 三、在远端实际使用（最小示例）

```bash
cd /path/to/your/repo
codex  # 进入交互式 TUI，让它读/改/跑当前目录代码
# 或一次性执行
codex exec "run tests and fix failing cases"
# 切换模型（如 gpt-5-codex）
codex --model gpt-5-codex
```

（Codex CLI 运行在当前工作目录的沙盒中；默认需批准越界操作或联网。审批模式可在 TUI 用 `/approvals` 切换。）([OpenAI开发者][1])

---

## 四、安全与常见坑

* **只在需要的目录运行**：Codex 仅访问当前目录，越界/联网需你确认。([OpenAI开发者][1])
* **凭证权限**：`~/.codex/auth.json` 建议 `chmod 600`，并限制服务器账号权限。([GitHub][2])
* **版本更新**：`npm install -g @openai/codex@latest` 或 `brew upgrade codex`（若用 Homebrew on Linux）。([OpenAI开发者][1])

---

## 参考（官方）

* Codex CLI 概览与安装（developers.openai.com）([OpenAI开发者][1])
* 开源仓库 README / 认证与无头登录入口（GitHub）([GitHub][2])

需要我给你一段 **一键脚本**（检测、安装、校验 auth.json、验证 `codex` 可用）吗？我可以直接贴出 bash 脚本。

[1]: https://developers.openai.com/codex/cli/ "Codex CLI"
[2]: https://github.com/openai/codex "GitHub - openai/codex: Lightweight coding agent that runs in your terminal"
