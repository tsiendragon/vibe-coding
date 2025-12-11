# Command 命令规范

## 1. 基本结构

### 文件格式
- **文件名**: `.claude/commands/{command-name}.md` (kebab-case)
- **编码**: UTF-8
- **格式**: Markdown

### 命令描述格式
```markdown
根据 $param1 和 $param2 执行特定操作，将结果输出到 $output_file
```

## 2. 命名规范

### 命令名称
- 使用 kebab-case 格式
- 动词开头，描述具体动作
- 避免过于通用的名称

### 示例
```
ask-mcq          # 生成选择题
grade-short      # 评分简答题
report-interview # 生成面试报告
validate-code    # 验证代码
deploy-service   # 部署服务
```

## 3. 参数规范

### 参数类型
- `$param`: 位置参数（按顺序传递）
- `${variable}`: 变量替换（来自工作流参数）
- `$input_file`: 输入文件路径
- `$output_file`: 输出文件路径

### 参数命名
- 使用 snake_case 格式
- 描述性命名，避免缩写
- 输入在前，输出在后

### 示例
```bash
/ask-mcq  ${topic}  ${id}  questions/mcq_q${id}.md  keys/mcq_q${id}.txt
/grade-short  ${id}  ${rubric}  scores/short_q${id}.json
/validate-code  src/main.py  reports/validation.md
```

## 4. 命令分类

### 按功能分类
```markdown
## 生成类命令 (Generate)
- ask-*: 生成问题类命令
- create-*: 创建文件/资源类命令
- build-*: 构建/编译类命令

## 处理类命令 (Process)
- grade-*: 评分/评估类命令
- validate-*: 验证/检查类命令
- transform-*: 转换/处理类命令

## 分析类命令 (Analyze)
- analyze-*: 分析类命令
- review-*: 审查类命令
- inspect-*: 检查类命令

## 报告类命令 (Report)
- report-*: 生成报告类命令
- summarize-*: 总结类命令
- export-*: 导出类命令

## 工具类命令 (Utility)
- record-*: 记录类命令
- cleanup-*: 清理类命令
- setup-*: 设置类命令
```

### 按执行环境分类
```markdown
## Chat模式命令
- 需要LLM推理和生成
- 复杂的文本处理
- 创意性任务

## Shell模式命令
- 文件系统操作
- 自动化脚本执行
- 系统命令调用

## API模式命令
- 外部服务调用
- 数据获取和提交
- 第三方集成
```

## 5. 执行模式

### Chat模式
```yaml
command:
  id: cmd.generate.content
  mode: chat
  agent: content_generator
  signature: "/generate-content ${topic} ${style} ${output_file}"
```

### Shell模式
```yaml
command:
  id: cmd.validate.syntax
  mode: shell
  agent: validator
  signature: "/validate-syntax ${source_file} ${report_file}"
```

### API模式
```yaml
command:
  id: cmd.fetch.data
  mode: api
  agent: data_fetcher
  signature: "/fetch-data ${endpoint} ${params} ${output_file}"
  config:
    endpoint: "https://api.example.com"
    auth_required: true
```

## 6. 输入输出规范

### 输入规范
```markdown
## 输入类型
- 文件路径: 相对于工作空间根目录
- 参数值: 字符串、数字、布尔值
- 配置对象: 结构化配置数据

## 输入验证
- 文件存在性检查
- 参数类型验证
- 权限检查
- 依赖项验证
```

### 输出规范
```markdown
## 输出类型
- 文件: 指定路径的文件输出
- 变量: 设置工作流变量
- 状态: 更新执行状态
- 日志: 记录执行信息

## 输出格式
- 文本文件: UTF-8编码，Unix换行
- JSON文件: 标准JSON格式，缩进2空格
- 报告文件: Markdown格式，标准结构
```