# Vibe Coding - AI协作开发框架

基于Claude Code的多Agent协作开发工作空间，支持PyTorch ML项目、Web开发、后端服务和Android开发。

## 核心特性

本框架实现了完整的AI Agent协作开发生态系统：
- 🤖 **7个专业AI Agent** - 产品经理、技术负责人、研究员、算法工程师、代码审查员、QA工程师、文档编写员
- 📋 **标准化工作流** - 需求分析→技术设计→原型开发→完整实现→测试验收→项目交付
- 📚 **完善的模板体系** - PRD、技术规格、原型设计、测试策略等40+模板
- 🔧 **严格的代码规范** - Python、PyTorch、Git、pytest等行业标准
- 🧠 **知识管理系统** - 最佳实践积累和错误案例库

## 快速开始

### 1. 创建新项目

使用项目设置脚本快速创建：

```bash
# 基本用法
./setup_project.sh <project_name> <project_type> [target_directory]

# 支持的项目类型示例
./setup_project.sh my-classifier pytorch      # PyTorch深度学习项目
./setup_project.sh web-app web               # Web应用项目  
./setup_project.sh data-analysis data-science # 数据科学项目
./setup_project.sh research-proj research     # 研究型项目
./setup_project.sh general-app general       # 通用软件项目

# 指定目标目录（例如PyTorch项目）
./setup_project.sh image-classifier pytorch ~/projects/
```

设置完成后将自动：
-  创建项目目录结构
-  复制文档模板到 `docs/templates/`
-  复制标准规范到 `docs/standards/` 
-  复制Agent配置到 `.claude/agents/`
-  复制工作流文档到 `docs/workflows/`
-  创建项目专用配置文件
-  生成项目README

### 2. 使用Claude Code Agents

项目包含预配置的Agent配置文件，位于 `.claude/agents/` 目录：

```
.claude/agents/
├── agent-product-manager.md     # 产品经理Agent
├── agent-tech-lead.md          # 技术负责人Agent  
├── agent-researcher.md         # 研究员Agent
├── agent-algorithm-engineer.md # 算法工程师Agent
├── agent-code-reviewer.md      # 代码审查员Agent
├── agent-qa-engineer.md        # QA测试Agent
└── agent-docs-writer.md        # 文档编写员Agent
```

在Claude Code中激活指定Agent：

```bash
# 启动产品经理Agent
claude-code --agent product-manager

# 启动技术负责人Agent  
claude-code --agent tech-lead

# 启动算法工程师Agent
claude-code --agent algorithm-engineer

# 查看所有可用Agent
claude-code --list-agents
```

### 3. 完整AI协作开发工作流

按照标准工作流阶段进行：

#### 阶段1: 需求分析
```bash
# 使用产品经理Agent编写PRD
claude-code --agent product-manager
> 请根据需求描述编写PRD文档，保存到docs/PRD.md
```

#### 阶段2: 技术设计  
```bash
# 启动技术负责人Agent进行架构设计
claude-code --agent tech-lead
> 基于PRD文档设计技术方案和架构

# 启动研究员Agent进行技术调研
claude-code --agent researcher  
> 调研相关技术方案和SOTA方法，生成可行性分析
```

#### 阶段3: 原型开发
```bash
# 使用算法工程师Agent进行核心算法实现
claude-code --agent algorithm-engineer
> 实现核心算法原型，进行验证实验

# 使用代码审查员Agent进行代码审查
claude-code --agent code-reviewer
> 审查代码质量和规范合规性

# 使用QA测试Agent进行测试
claude-code --agent qa-engineer  
> 为原型编写测试用例，进行集成测试和性能测试
```

#### 阶段4: 文档和交付
```bash
# 使用文档编写员Agent完善文档
claude-code --agent docs-writer
> 生成项目文档、用户指南和API文档

# 使用技术负责人Agent进行最终验收
claude-code --agent tech-lead
> 进行最终验收和交付决策
```

## Agent角色详解

### 🎯 Product Manager (agent-product-manager)
**核心职责**: 需求分析、PRD编写、功能验收
- 编写详细需求文档 (`docs/PRD.md`)
- 功能验收和用户体验设计
- 需求管理和优先级排序

**主要输出**: PRD.md、需求验收报告
**关键阶段**: 项目启动、需求确认、最终验收

### 🏗️ Tech Lead (agent-tech-lead)  
**核心职责**: 技术方案设计、架构决策、项目协调
- 制定技术架构和设计方案 (`docs/TECH_SPEC.md`)
- 管理项目任务清单 (`docs/TODO.md`) 
- 协调各Agent的协作配合
- 最终交付决策和质量把关

**主要输出**: TECH_SPEC.md、TODO.md、项目验收报告
**关键阶段**: 架构设计、里程碑决策、项目验收

### 🔬 Researcher (agent-researcher)
**核心职责**: 技术调研、可行性分析  
- 编写文献综述和SOTA分析 (`docs/research/literature_review.md`)
- 编写技术方案建议 (`docs/research/recommendations.md`)
- 制定理论一致性审核标准

**主要输出**: literature_review.md、recommendations.md、理论审核报告
**关键阶段**: 技术调研、理论验证、算法审核

### ⚙️ Algorithm Engineer (agent-algorithm-engineer)
**核心职责**: 算法实现、模型设计、核心开发
- 实现核心算法和模型架构
- 编写验证实验和性能测试  
- 编写模块级README和技术文档
- 制定实现方案和代码开发

**主要输出**: 核心算法代码、PROTOTYPE.md、验证报告
**关键阶段**: 算法实现、模块开发、性能优化

### 🔍 Code Reviewer (agent-code-reviewer)
**核心职责**: 代码质量审核、标准检查、持续监控
- 审查代码质量和编程规范
- 检查PyTorch最佳实践合规性
- 编写代码审核报告和改进建议
- 推广编程最佳实践

**主要输出**: 代码审核报告、质量改进建议
**关键阶段**: 代码开发全程、最终代码审核

### 🧪 QA Engineer (agent-qa-engineer)  
**核心职责**: 测试用例编写、质量保证、性能测试
- 编写单元测试、集成测试、E2E测试
- 进行性能基准测试和鲁棒性测试
- 编写测试报告和质量评估
- 制定项目质量验收标准

**主要输出**: 测试代码、测试报告、质量评估报告  
**关键阶段**: 模块开发、集成测试、最终验收

### 📝 Docs Writer (agent-docs-writer)
**核心职责**: 技术文档、项目文档、文档体系构建
- 编写API文档和用户指南
- 整理文档体系和知识管理  
- 编写最终项目README和说明文档
- 推广文档协作模式

**主要输出**: README.md、API文档、用户指南
**关键阶段**: 最终文档生成

## 工作流阶段

完整的AI协作开发流程包含6个主要阶段：

### 阶段1: 需求分析阶段
1. **Product Manager** 编写PRD文档
2. **Tech Lead** 审核需求可行性
3. 确认项目边界和验收标准

### 阶段2: 技术设计阶段
1. **Researcher** 进行技术调研和可行性分析
2. **Tech Lead** 设计技术架构和实现方案
3. **多Agent评审** TECH_SPEC技术规格
4. **Tech Lead** 生成项目任务清单

### 阶段3: 原型开发阶段  
1. **Algorithm Engineer** 实现核心算法原型
2. **QA Engineer** 编写原型测试
3. **三方评估** (Tech Lead + QA + Researcher)
4. 确定原型是否满足要求

### 阶段4: 完整开发阶段
1. **Algorithm Engineer** 模块化开发
2. **Code Reviewer** 持续代码审查  
3. **QA Engineer** 编写完整测试
4. 迭代开发直到功能完整

### 阶段5: 测试验证阶段
1. **QA Engineer** 集成测试、性能测试、E2E测试
2. 性能基准对比测试
3. 鲁棒性测试和稳定性验证
4. 生成测试报告

### 阶段6: 项目验收阶段
1. **Code Reviewer** 最终代码审核
2. **Tech Lead** 架构质量评估  
3. **Product Manager** 需求验收
4. **QA Engineer** 质量验收
5. **Researcher** 理论一致性审核
6. **Tech Lead** 最终交付决策

## 模板体系

项目提供40+标准化模板，覆盖开发全流程：

### 需求分析模板
- `docs/templates/PRD/prd_template.md` - 产品需求文档
- `docs/templates/TECH_SPEC/TECH_SPEC_template.md` - 技术规格文档  
- `docs/templates/TODO/TODO_template.md` - 任务管理
- `docs/templates/TODO/project_todo_template.md` - 项目TODO

### 研究调研模板
- `docs/templates/research/literature_review_template.md` - 文献综述
- `docs/templates/research/feasibility_analysis_template.md` - 可行性分析
- `docs/templates/research/recommendations_template.md` - 技术建议

### 开发测试模板
- `docs/templates/PROTOTYPE/PROTOTYPE_template.md` - 原型设计
- `docs/templates/tests/integration_report_template.md` - 集成测试报告
- `docs/templates/tests/e2e_report_template.md` - E2E测试报告
- `docs/templates/performance_test_template.md` - 性能测试
- `docs/templates/test_strategy_template.md` - 测试策略

### 文档交付模板
- `docs/templates/README_template.md` - 项目README
- `docs/templates/API_docs_template.md` - API文档  
- `docs/templates/user_guide_template.md` - 用户指南
- `docs/documentation/project_acceptance_report_template.md` - 项目验收报告

## 标准规范

### Python代码规范 (`docs/standards/pycode_standards.md`)
- PEP8编码标准
- 类型注解要求
- 异常处理规范
- 性能最佳实践

### PyTorch开发规范 (`docs/standards/pytorch_standards.md`)  
- 项目结构标准
- Lightning框架规范
- 模型架构最佳实践
- 配置管理模式

### 测试规范 (`docs/standards/pytest_stands.md`)
- 分层测试体系 (unit/integration/e2e)
- 90%测试覆盖率要求
- 测试命名和组织规范
- 性能测试和鲁棒性测试

### Git提交规范 (`docs/standards/git_commit_std.md`)
- 规范提交格式
- 类型和作用域定义
- 提交内容要求
- 分支管理策略

## 知识管理

### 最佳实践库 (`docs/knowledge/best_practices/`)
- `tech_solutions.md` - 技术方案最佳实践
- `code_patterns.md` - 代码模式库
- `test_strategies.md` - 测试策略集锦  
- `collaboration_patterns.md` - Agent协作模式

### 错误案例库 (`docs/knowledge/error_cases/`)
- `common_issues.md` - 常见问题解决
- 调试方法集合
- 性能问题解决方案

## 实战案例

### 示例: PyTorch图像分类项目

```bash
# 1. 创建项目
./setup_project.sh image-classifier pytorch

# 2. 进入项目目录
cd image-classifier

# 3. 需求分析
claude-code --agent product-manager
>  请为图像分类项目编写PRD，要求支持10个类别，准确率>90%

# 4. 技术调研
claude-code --agent researcher  
> 调研图像分类SOTA方法，推荐最佳算法方案

# 5. 架构设计
claude-code --agent tech-lead
> 基于调研结果设计技术方案，包括模型架构和数据流

# 6. 算法实现
claude-code --agent algorithm-engineer
> 实现ResNet50图像分类模型，包括数据加载和训练流程

# 7. 代码审查
claude-code --agent code-reviewer
> 审查代码质量和PyTorch最佳实践

# 8. 测试验证  
claude-code --agent qa-engineer
> 编写完整测试用例，进行集成测试和性能测试

# 9. 文档整理
claude-code --agent docs-writer
> 生成项目文档、README和API文档说明

# 10. 项目验收
claude-code --agent tech-lead
> 进行最终验收和交付决策
```

### Web应用开发项目

```bash
# 1. 创建React项目
./setup_project.sh user-dashboard web

# 2. 使用产品经理Agent进行需求分析
claude-code --agent product-manager
> 为用户Dashboard编写PRD，包括用户管理、数据可视化功能

# 3. 按照工作流继续后续阶段...
```

## 常见问题

### 基础问题

**Q: Agent配置文件在哪里？**
A: 所有Agent配置文件位于 `.claude/agents/` 目录，文件名格式: `agent-*.md`

**Q: 文档模板如何使用？**  
A: 查看 `docs/templates/` 目录下的各种模板，或使用 `setup_project.sh` 自动复制

**Q: 代码规范是什么？**
A: 查看 `docs/standards/` 目录的详细规范文档，包括编码和测试标准

**Q: 测试覆盖率如何查看？**
A: 查看 `docs/standards/pytest_stands.md` 的测试标准，使用 `pytest --cov` 检查覆盖率

### 进阶指导

- 完整工作流: 查看 `docs/workflows/workflow.md` 了解详细工作流程
- 知识管理: 查看 `docs/knowledge/` 了解最佳实践和错误案例
- 模板定制: 参考现有模板进行项目定制
- 问题追踪: 使用项目 `docs/TODO.md` 进行任务管理

## 发展路线图

1. **持续Agent协作优化** - 优化Agent间的协作效率
2. **知识库扩充** - 积累更多项目经验和最佳实践到knowledge目录
3. **模板体系完善** - 增加更多项目类型和场景模板
4. **自动化提升** - 持续改进自动化工具和检查

## 开源协议

MIT License - 详见 LICENSE 文件

---

**通过AI协作开发，让软件工程更智能、更高效** 

*多Agent协作，AI驱动的下一代开发范式*