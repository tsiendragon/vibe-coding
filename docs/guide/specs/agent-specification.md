# Agent 智能体规范文档

定义Claude Code智能体的标准结构和行为规范，确保智能体的一致性和可维护性。

## 1. 文件结构

**文件命名**: `./claude/agents/agent-name.md` (使用短横线连接)
**文档格式**: Markdown + YAML元数据

### 元数据定义
```yaml
---
name: agent_name                    # 智能体唯一标识
description: "智能体功能说明"        # 一句话描述功能
version: "1.0.0"                   # 语义化版本号
tools: [Read, Write, Shell]        # 授权工具列表
model: sonnet                      # 推荐AI模型
color: teal                        # 界面显示颜色
category: "functionality"          # 功能分类
tags: ["tag1", "tag2"]            # 关键词标签
---
```

## 2. 智能体职责契约

明确定义智能体的工作范围和职责边界。

```markdown
## contract
- goal: 智能体的核心使命（一句话说明）
- scope: 工作范围和处理的任务类型
- responsibilities: 具体负责的工作内容
  - 主要职责1
  - 主要职责2
- limitations: 明确不负责的工作
  - 不处理的任务类型
  - 权限限制
```

### 依赖声明
```markdown
- relies_on:
  - .claude/standards/coding-standards.md  # 编码规范
  - .claude/commands/test-runner          # 测试命令
  - external-api                         # 外部API
```

## 3. 输入输出接口

定义智能体与外部系统的交互接口。

### 输入参数
```markdown
## inputs
- required: 必需参数
  - param_name (类型): 参数说明
  - config_file (string): 配置文件路径
  - data_source (object): 数据源

- optional: 可选参数
  - debug_mode (boolean): 调试模式，默认false
  - timeout (number): 超时时间（秒），默认300

- context: 上下文信息
  - workflow_state: 当前工作流状态
  - previous_outputs: 前序步骤输出
```

### 输出结果
```markdown
## outputs
- files: 生成的文件
  - reports/analysis.md: 分析报告
  - artifacts/data.json: 结构化数据
  - logs/execution.log: 执行日志

- variables: 返回变量
  - status (string): 执行状态
  - score (number): 评分结果
  - recommendations (array): 建议列表

- state_changes: 状态更新
  - workflow_progress: 工作流进度
  - agent_memory: 智能体记忆
```

## 4. 行为逻辑

定义智能体的核心动作和决策逻辑。

### 核心动作
```markdown
## actions
- primary: 主要执行动作
  - action_name: `/command-name $param1 $param2`
  - description: 动作说明和触发条件

- secondary: 辅助动作
  - validation: 输入验证
  - error_handling: 异常处理

- maintenance: 维护动作
  - cleanup: 资源清理
  - logging: 过程记录
```

### 决策规则
```markdown
## decision_logic
- conditions: 条件分支
  - if input_valid: execute_main_flow
  - if error_detected: execute_error_flow
  - if resource_unavailable: wait_and_retry

- strategies: 执行策略
  - performance_mode: 优先效率
  - quality_mode: 优先质量
  - balanced_mode: 效率质量平衡
```


## 5. 安全规范

确保智能体运行的安全性和合规性。

```markdown
## security
- access_control: 访问控制
  - authentication_required: 是否需要认证
  - authorization_levels: 权限级别
  - permission_boundaries: 权限范围

- data_protection: 数据保护
  - sensitive_data_handling: 敏感信息处理
  - encryption_requirements: 加密需求
  - data_retention_policy: 数据保留规则

- audit_requirements: 审计需求
  - action_logging: 操作记录
  - access_tracking: 访问跟踪
  - compliance_monitoring: 合规监控
```


## 6. 版本控制（可选）

对需要版本管理的智能体，遵循以下规范：

- **版本格式**: 语义化版本 MAJOR.MINOR.PATCH
- **兼容性**: 保持向后兼容
- **升级指南**: 提供版本迁移说明
- **变更记录**: 记录每版本的功能变更

## 7. 设计原则

智能体设计应遵循以下最佳实践：

1. **单一职责** - 每个智能体专注特定领域
2. **接口清晰** - 明确输入输出规范
3. **错误处理** - 完善的异常处理机制
4. **文档完整** - 详细的使用说明
5. **安全优先** - 考虑数据安全和隐私
6. **性能优化** - 高效的资源使用
7. **易于测试** - 支持完整的测试覆盖
