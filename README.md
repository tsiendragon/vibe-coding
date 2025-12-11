# Vibe Coding - AI驱动的自动化工作流框架

> 基于 Claude Code CLI 的自动化工作流编排框架

## 🏛️ 整体架构

### 📁 仓库结构

```
vibe-coding/
├── 📊 docs/                           # 文档中心
│   ├── 📚 claude-code/                # Claude Code CLI 相关文档
│   │   ├── introduction.md           # Claude Code 功能介绍
│   │   ├── automation-research.md    # 自动化框架研究
│   │   └── images/                   # Claude Code 相关图片
│   ├── 🎯 project/                    # Vibe Coding 项目文档
│   │   ├── specs/                    # 组件规范文档
│   │   ├── examples/                 # 项目使用示例
│   │   ├── framework-spec.md         # 自动化框架规范
│   │   ├── repository-structure.md  # 仓库结构说明
│   │   └── images/                   # 项目相关图片
│   └── 🖼️ shared/                     # 共享资源
│       └── images/                   # 通用图片资源
├──
├── 🎨 projects/                       # 📋 项目模板库
│   ├── 📱 android/                    # Android/Flutter 项目
│   │   ├── agents/                   # Android专用代理
│   │   ├── standards/                # Android开发标准
│   │   ├── workflows/                # Android开发流程
│   │   ├── templates/                # Android文档模板
│   │   └── config.yaml              # 项目配置文件
│   ├── 🚀 backend/                    # 后端API项目
│   │   ├── agents/                   # 后端专用代理
│   │   ├── standards/                # 后端开发标准
│   │   ├── workflows/                # 后端开发流程
│   │   ├── templates/                # 后端文档模板
│   │   └── config.yaml              # 项目配置文件
│   ├── 🧠 pytorch/                    # PyTorch深度学习项目
│   │   ├── agents/                   # PyTorch专用代理
│   │   ├── standards/                # PyTorch开发标准
│   │   ├── workflows/                # PyTorch开发流程
│   │   ├── templates/                # PyTorch文档模板
│   │   └── config.yaml              # 项目配置文件
│   └── 🌐 web/                        # Web前端项目
│       ├── agents/                   # Web专用代理
│       ├── standards/                # Web开发标准
│       ├── workflows/                # Web开发流程
│       ├── templates/                # Web文档模板
│       └── config.yaml              # 项目配置文件
├──
├── 📚 docs/                           # 🔍 完整文档中心
│   ├── 🤖 claude-code/                # Claude Code CLI 专题文档
│   │   ├── introduction.md           # Claude Code CLI 完整使用指南
│   │   └── automation-research.md    # 自动化工作流框架调研报告
│   ├── 📋 guide/                      # 框架使用指南
│   │   ├── specs/                    # 核心规范文档
│   │   │   ├── agent-specification.md      # Agent 智能体规范
│   │   │   ├── command-specification.md    # Command 命令规范
│   │   │   ├── standards-specification.md  # Standards 标准规范
│   │   │   ├── workflow-specification.md   # Workflow 工作流规范
│   │   │   └── template-specification.md   # Template 模板规范
│   │   └── examples/                 # 实践示例
│   │       └── interview_workflow_example.md # 自动化面试工作流示例
│   └── 🖼️ images/                     # 文档配图资源
├──
├── 🌐 common/                         # 🔗 通用组件库
│   ├── commands/                     # 通用命令库
│   └── shared/                       # 共享资源
├──
├── 🎯 examples/                       # 📌 参考示例项目
│   ├── interview/                    # 面试自动化示例
│   │   ├── .claude/
│   │   │   ├── workflow/workflow.yaml # 面试专用工作流
│   │   │   ├── agents/               # 面试专用代理
│   │   │   └── standards/            # 面试质量标准
│   │   └── claude.md
│   └── story_tell/                   # 内容创作示例
├── 🛠️ script/                          # 工具脚本
└── 🏗️ setup_project.sh                # 项目初始化脚本 (支持config配置)
```

## 📚 文档导航

### 🤖 Claude Code CLI 专题文档

深入了解 Claude Code CLI 的功能和自动化工作流应用：

- **[Claude Code CLI 完整使用指南](docs/claude-code/introduction.md)**
  - 从零到精通 Claude Code CLI：记忆管理、子 Agent、权限控制、工作流与实战技巧
  - 涵盖记忆系统、自定义命令、MCP 集成、Python 集成、GitHub 集成等核心功能
  - 与 Cursor 的对比分析，了解为什么选择命令行 AI 工具

- **[自动化工作流框架调研报告](docs/claude-code/automation-research.md)**
  - Claude Code CLI 在 DevOps/MLOps 自动化流程中的应用实践
  - 社区生态系统分析：工具链、项目模板、最佳实践案例
  - "Vibe-coding" 范式解析：以自然语言为主的编程方式

### 📋 框架规范文档

掌握 Vibe Coding 框架的核心设计理念和技术规范：

#### 🔧 核心规范

- **[Agent 智能体规范](docs/guide/specs/agent-specification.md)**
  - 定义 Claude Code 智能体的标准结构和行为规范
  - 涵盖元数据定义、职责契约、输入输出接口、行为逻辑、安全规范

- **[Command 命令规范](docs/guide/specs/command-specification.md)**
  - 自定义命令的标准化设计规范
  - 包含命名规范、参数规范、执行模式、输入输出规范

- **[Standards 标准规范](docs/guide/specs/standards-specification.md)**
  - 各类标准文档的结构和分类体系
  - 定义强制性标准、推荐标准、可选标准的执行机制

- **[Workflow 工作流规范](docs/guide/specs/workflow-specification.md)**
  - 工作流 YAML 配置的完整规范定义
  - 涵盖参数配置、引用注册表、阶段步骤、错误处理等核心概念

- **[Template 模板规范](docs/guide/specs/template-specification.md)**
  - 模板系统的设计原则和使用规范
  - 包含文档模板、报告模板、流程模板等多种类型

#### 🎯 实践示例

- **[自动化面试工作流示例](docs/guide/examples/interview_workflow_example.md)**
  - 完整的技术面试自动化流程演示
  - Multi-Agent 架构：面试官、记录员、评估员协同工作
  - 支持选择题、填空题、简答题的混合面试模式
  - 智能评分系统和详细反馈报告生成

### 🚀 快速开始

1. **新手入门**：建议先阅读 [Claude Code CLI 完整使用指南](docs/claude-code/introduction.md)
2. **框架理解**：通过 [自动化工作流框架调研报告](docs/claude-code/automation-research.md) 了解技术背景
3. **实践操作**：参考 [自动化面试工作流示例](docs/guide/examples/interview_workflow_example.md) 进行实战练习
4. **深入定制**：根据需求查阅相应的规范文档进行项目配置和扩展
