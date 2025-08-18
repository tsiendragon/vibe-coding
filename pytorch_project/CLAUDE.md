# PyTorch 深度学习项目

基于 PyTorch 的深度学习项目，采用 AI 多智能体协作开发模式。

## 🚀 项目启动

### 明确需求
**必须先明确项目需求**

**需求来源：**
- 已有需求文档 - 请提供文档路径或内容
- 需要讨论需求 - 我们对话澄清
- 要实现论文 - 请提供论文或描述模型
- 需要反复跟用户讨论 - 明确需求

**重要**：Product Manager agent 会确保需求100%明确后才开始开发

### AI Agent 团队
- **🎯 产品经理** - 需求澄清，编写PRD
- **🏗️ 技术主管** - 架构设计，协调开发
- **🔬 研究员** - 文献分析，理论指导
- **⚡ 算法工程师** - 模型实现，生产级代码
- **🔍 代码审查员** - 质量把控，规范检查
- **🧪 QA工程师** - 测试验证，质量保证
- **📚 文档工程师** - 文档编写，知识沉淀

## 📋 必读文档

**开发前必须理解以下文档：**

- **`docs/workflows/workflow.md`** - 完整工作流程，包含详细的阶段任务、Git提交时机、知识沉淀要求
- **`.claude/agents/`** - 各agent职责和协作模式
- **`docs/standards/`** - 开发规范（Python、PyTorch、测试、Git提交）
- **`docs/templates/`** - 文档模板（PRD、TECH_SPEC、PROTOTYPE、测试等）
- **`docs/knowledge/`** - 最佳实践和常见问题解决方案

## ⚠️ 关键要求

### 工作流遵循
**严格按照 `docs/workflows/workflow.md` 执行**：

### Git提交要求
**必须按照workflow规定的时机进行commit**：

- Source Inventory 的

### 质量标准
- **禁止dummy函数** - 使用 `raise NotImplementedError()`
- **禁止print语句** - 使用logging
- **90%测试覆盖率** - 关键功能必须测试
- **每个模块完成时commit** - 严格执行增量开发
## MUST
1. **Source Inventory** 中每条资源**必须**附上可访问的网址链接。
2. 进入下一步前，先阅读 `docs/workflow/workflow.md`，核对是否**严格**按流程执行；若偏离，**立即纠偏**回到正轨。
3. 产出 `docs/research/concept_inventory.md` 后，与用户**确认已理解/未理解**项；依据用户反馈，对**不清楚或部分理解**的概念继续深挖研究。
4. 对具体概念形成理论理解时，**同步拉取实现代码进行验证**；若实现与理论不一致，**立即反思并校正理论**，从多角度交叉验证其正确性。
5. 理解具体概念时，**拆解并逐一确认其子概念**是否已被充分理解。
6. agent-researcher对概念进行验证时，拉取的代码也保存下来，为后续参考。保存原始代码 不要保存加工后的代码
7. agent-researcher 验证之前的理解错误时，要及时纠正之前错误的文档

8. 下载 GitHub 原始文件时,原始网址是预览网址，需要切换成实际网址
例如预览: https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/fp8.py
原始: https://raw.githubusercontent.com/NVIDIA/TransformerEngine/main/transformer_engine/pytorch/fp8.py.需要进行转换
9. 调用agent-tech-lead来详细设计子模块，对每个模块进行深度拆分，直到单个模块足够简单且职责单一
**请提供项目需求或说明想要构建什么，Product Manager agent 将主导需求澄清过程。**



