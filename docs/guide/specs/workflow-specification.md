# Workflow 工作流规范

## 1. 基本结构

### 文件格式
- **文件名**: `workflow.yaml`
- **编码**: UTF-8
- **格式**: YAML 1.2

### 基本元信息
```yaml
apiVersion: wop/v0.3           # 工作流协议版本
kind: Workflow                 # 固定值：Workflow
metadata:
  name: workflow-name          # 工作流名称（kebab-case）
  version: 1.0.0              # 语义化版本号
  description: "工作流描述"     # 可选的描述信息
  tags: ["tag1", "tag2"]       # 可选的标签
```

## 2. 参数配置

### 全局参数
```yaml
params:
  # 基础参数
  param_name: "default_value"
  numeric_param: 42
  boolean_param: true

  # 复杂参数（支持多行字符串）
  complex_param: |
    多行字符串内容
    支持换行和格式化
```

### 参数类型支持
- `string`: 字符串类型
- `number`: 数值类型
- `boolean`: 布尔类型
- `array`: 数组类型
- `object`: 对象类型

## 3. 引用注册表

### 标准规范引用
```yaml
refs:
  standards:
    - id: std.unique.identifier    # 标准唯一标识
      name: "标准名称"              # 人类可读名称
      uri: ".claude/standards/file.md"  # 文件路径
      enforce:                     # 执行规则
        when: ["step1", "step2"]   # 在哪些步骤执行
        checks:                    # 检查规则
          - type: tool             # 检查类型
            tool: "@tool.name"     # 工具引用
            args: ["arg1", "arg2"] # 工具参数
```

### 命令注册
```yaml
  commands:
    - id: cmd.unique.identifier    # 命令唯一标识
      agent: agent_name            # 执行代理
      mode: chat|shell             # 执行模式
      signature: "/cmd-name $param1 $param2 $output" # 命令签名
      outputs:                     # 输出定义
        files: ["output1.txt"]     # 输出文件列表
        variables: ["var1"]        # 输出变量列表
```

### 工具注册
```yaml
  tools:
    - id: tool.unique.identifier   # 工具唯一标识
      kind: shell|api|claude       # 工具类型
      entry: "command or endpoint" # 入口点
      config:                      # 配置参数
        timeout: 30                # 超时时间（秒）
        retry: 3                   # 重试次数
```

## 4. 产物定义

```yaml
artifacts:
  # 定义工作流产物
  main_output: "path/to/output.md"     # 主要输出
  summary_json: "artifacts/summary.json" # 汇总数据
  final_report: "reports/final.md"     # 最终报告
```

## 5. 代理配置

```yaml
agents:
  - id: main_agent               # 代理ID
    role: HostAgent             # 代理角色
    model: sonnet               # 可选：指定模型
    config:                     # 可选：代理配置
      temperature: 0.7
      max_tokens: 4000
```

## 6. 阶段定义

```yaml
stages:
  - id: preparation              # 阶段ID
    name: "准备阶段"             # 阶段名称
    description: "阶段描述"       # 可选描述
    steps: [step1, step2]        # 包含的步骤

  - id: execution
    name: "执行阶段"
    steps: [step3, step4]
    depends_on: [preparation]    # 可选：依赖的阶段
```

## 7. 步骤定义

### 基本步骤
```yaml
steps:
  - id: step_unique_id           # 步骤唯一标识
    name: "步骤名称"             # 人类可读名称
    agent: agent_name            # 执行代理
    description: "步骤描述"       # 可选描述

    # 执行定义
    run:
      type: chat|command|shell   # 执行类型
      command: "command to run"  # 执行的命令

    # 流程控制
    next: [next_step_id]         # 下一步骤
    condition: "expression"      # 可选：执行条件

    # 质量检查
    use:
      standards: ["std.id"]      # 使用的标准
```

### 循环步骤
```yaml
  - id: loop_step
    name: "循环步骤"
    agent: agent_name
    loop:
      mode: foreach|while        # 循环模式
      generator:                 # 循环生成器
        fn: range                # 函数：range|list|custom
        start: 1                 # 开始值
        end: "${params.count}"   # 结束值
      body:                      # 循环体
        type: composite          # 复合步骤
        steps:
          - type: command
            command: "/cmd ${loop.value}"
          - type: gate           # 网关步骤
            gate:
              type: approval     # 网关类型
              id: "gate_id"      # 网关ID
              approvers: ["用户"] # 审批者
              prompt: "审批提示" # 审批提示
```

### 网关步骤
```yaml
gates:
  - type: approval               # 审批网关
    id: gate_unique_id          # 网关ID
    approvers: ["user", "role"] # 审批者列表
    prompt: "审批提示信息"       # 提示信息
    timeout: 3600               # 可选：超时时间（秒）

  - type: condition             # 条件网关
    id: condition_gate
    condition: "${var} > 0"     # 条件表达式

  - type: manual                # 手动网关
    id: manual_gate
    instruction: "手动操作说明"  # 操作说明
```

## 8. 变量和表达式

### 变量引用
- 参数引用: `${params.param_name}`
- 输出引用: `${artifacts.output_name}`
- 循环变量: `${loop.value}`, `${loop.index}`
- 网关结果: `${gate('gate_id').result}`

### 条件表达式
- 比较: `${var} > 10`, `${var} == 'value'`
- 逻辑: `${var1} && ${var2}`, `${var1} || ${var2}`
- 函数: `exists(${file_path})`, `length(${array}) > 0`

## 9. 错误处理

```yaml
steps:
  - id: error_prone_step
    name: "可能出错的步骤"
    run:
      type: command
      command: "/risky-command"
    error_handling:
      retry: 3                   # 重试次数
      on_failure:               # 失败时的处理
        - type: command
          command: "/cleanup"
        - type: notify
          message: "步骤失败通知"
```

## 10. 版本控制

### 版本兼容性
- 主版本：不兼容的API变更
- 次版本：向后兼容的功能增加
- 修订版本：向后兼容的错误修复

### 迁移指南
当工作流规范版本升级时，提供自动化迁移工具支持。

## 11. 最佳实践

1. **命名规范**: 使用 kebab-case 命名
2. **参数验证**: 在工作流开始时验证所有必需参数
3. **错误恢复**: 为关键步骤提供错误恢复机制
4. **日志记录**: 记录关键步骤的执行状态
5. **资源清理**: 确保临时资源得到适当清理
