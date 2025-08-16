# PyTorch 深度学习项目

基于 PyTorch 的深度学习项目，采用 AI 多智能体协作开发模式。

## 🚀 项目启动指南

### 第一步：明确需求
**开发前必须先明确项目需求。**

**需求来源：**
- **方式A**：已有需求文档 - 请提供文档路径或内容
- **方式B**：需要讨论需求 - 我们对话澄清
- **方式C**：要实现论文 - 请提供论文或描述模型

**重要**：Product Manager agent 会确保需求100%明确后才开始开发。

### 第二步：AI Agent 团队
多智能体协作系统：

1. **🎯 产品经理** - 需求澄清，编写PRD
2. **🏗️ 技术主管** - 架构设计，协调开发  
3. **🔬 研究员** - 文献分析，理论指导
4. **⚡ 算法工程师** - 模型实现，生产级代码
5. **🔍 代码审查员** - 质量把控，规范检查
6. **🧪 QA工程师** - 测试验证，质量保证
7. **📚 文档工程师** - 文档编写，知识沉淀

## 📋 必读文档

**开发前必须理解以下关键文档：**

### 工作流程
- **`docs/workflows/workflow.md`** - AI协作开发完整流程
  - 了解agent协作方式、阶段转换、交付物

### Agent职责  
- **`.claude/agents/`** - 各agent角色和职责
  - 确保任务分配正确、协作模式清晰

### 开发规范
- **`docs/standards/pycode_standards.md`** - Python编码规范（禁止dummy函数、使用logging、快速失败）
- **`docs/standards/pytorch_standards.md`** - PyTorch开发规范
- **`docs/standards/pytest_stands.md`** - 测试规范（90%覆盖率）
- **`docs/standards/git_commit_std.md`** - Git提交规范

### 文档模板
- **`docs/templates/PRD/`** - 产品需求文档模板
- **`docs/templates/TECH_SPEC/`** - 技术规格书模板
- **`docs/templates/PROTOTYPE/`** - 原型开发模板
- **`docs/templates/tests/`** - 测试文档模板

### 知识管理
- **`docs/knowledge/best_practices/`** - 最佳实践积累
- **`docs/knowledge/error_cases/`** - 常见问题及解决方案

## 🎯 开发命令

### 环境与训练
```bash
# 安装依赖
pip install -r requirements.txt

# 默认配置训练
python -m src.train --config configs/default.yaml

# 自定义参数训练
python -m src.train --config configs/default.yaml \
  trainer.max_epochs=100 \
  trainer.devices=2 \
  data.params.batch_size=128
```

### 测试与质量
```bash
# 运行测试（含覆盖率）
pytest --cov=src --cov-report=term-missing

# 代码质量检查
black .                    # 格式化
ruff check . --fix        # 检查修复
mypy .                    # 类型检查

# 一键质量检查
black . && ruff check . && mypy .
```

## 🔄 AI开发流程

### 阶段1：需求与PRD（产品经理主导）
1. **客户需求确认** - 反复澄清直到100%明确
2. **PRD创建** - 结构化需求文档
3. **团队对齐** - 技术主管和QA审核

### 阶段2：研究与设计（研究员+技术主管）
1. **文献调研** - 研究现有方案
2. **技术规格** - 详细架构和实现方案
3. **可行性验证** - 确保技术和资源可行

### 阶段3：原型开发（算法工程师主导）
1. **核心组件实现** - 关键算法和模型
2. **原型验证** - 多agent评估
3. **架构优化** - 基于原型反馈改进

### 阶段4：完整实现（全员协作）
1. **模块开发** - 完整实现，持续审查
2. **测试集成** - QA确保每步质量
3. **性能优化** - 算法工程师优化生产代码

### 阶段5：验证与交付（技术主管协调）
1. **全面测试** - 所有质量关卡必须通过
2. **文档完善** - 用户指南和技术文档
3. **多方验收** - 所有agent验证交付物

## ⚠️ 关键原则

### Claude Code须知
1. **先明确需求再编码** - 必须先与产品经理确认需求
2. **严格遵循工作流** - 每阶段有特定交付物
3. **保持质量标准** - 代码必须通过所有检查
4. **正确协调agent** - 关键节点通知相关agent
5. **记录一切** - 每阶段产出特定文档

### 开发标准
- ✅ **禁止dummy函数** - 使用 `raise NotImplementedError()`
- ✅ **禁止print语句** - 使用logging
- ✅ **快速失败** - 最小化try/except  
- ✅ **90%测试覆盖率** - 关键功能必须测试
- ✅ **模块级提交** - 增量开发和版本控制

## 🤝 开始项目

1. **说明项目目标**：要构建什么？
2. **提供需求**：需求文档、论文或对话讨论
3. **激活agent**：产品经理主导需求澄清
4. **遵循流程**：各agent按序贡献专业能力
5. **质量交付**：获得生产级PyTorch实现

**准备好了吗？请提供项目需求或说明想要构建什么！**