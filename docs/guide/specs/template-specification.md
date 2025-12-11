# Template 模板规范

## 1. 基本结构

### 文件格式
- **文件名**: `{template-name}_template.md` (snake_case)
- **编码**: UTF-8
- **格式**: Markdown + 变量占位符

### 模板元信息
```yaml
---
template:
  name: "模板名称"
  version: "1.0.0"
  description: "模板描述"
  category: "模板分类"
  author: "作者信息"
  created: "2024-01-01"
  updated: "2024-01-01"
  tags: ["tag1", "tag2"]
variables:
  - name: "variable_name"
    type: "string|number|boolean|array|object"
    required: true|false
    default: "default_value"
    description: "变量描述"
---
```

## 2. 模板分类

### 按用途分类
```markdown
## 文档模板 (Documentation Templates)
- 技术规格模板: TECH_SPEC_template.md
- API文档模板: API_docs_template.md
- 用户指南模板: user_guide_template.md
- 发布说明模板: release_notes_template.md

## 报告模板 (Report Templates)
- 测试报告模板: test_report_template.md
- 性能报告模板: performance_report_template.md
- 质量报告模板: quality_report_template.md
- 项目验收模板: acceptance_report_template.md

## 流程模板 (Process Templates)
- 工作流模板: workflow_template.yaml
- 代理模板: agent_template.md
- 标准模板: standard_template.md
- 命令模板: command_template.md

## 配置模板 (Configuration Templates)
- 项目配置模板: project_config_template.yaml
- 部署配置模板: deployment_config_template.yaml
- 环境配置模板: environment_template.env

## 代码模板 (Code Templates)
- 类定义模板: class_template.py
- 接口模板: interface_template.ts
- 测试模板: test_template.py
```

### 按复杂度分类
```markdown
## 简单模板 (Simple Templates)
- 变量替换
- 基本结构生成
- 静态内容为主

## 中等模板 (Medium Templates)
- 条件逻辑
- 循环结构
- 计算字段

## 复杂模板 (Complex Templates)
- 嵌套模板
- 动态结构
- 外部数据集成
```

## 3. 变量系统

### 变量类型
```markdown
## 基础类型
- string: 字符串类型
- number: 数字类型
- boolean: 布尔类型
- date: 日期类型
- url: URL类型
- email: 邮箱类型

## 复合类型
- array: 数组类型
- object: 对象类型
- enum: 枚举类型

## 特殊类型
- file_path: 文件路径
- template_ref: 模板引用
- computed: 计算字段
```

### 变量定义
```yaml
variables:
  # 基础变量
  - name: project_name
    type: string
    required: true
    description: "项目名称"
    validation: "^[a-zA-Z][a-zA-Z0-9_-]*$"

  # 枚举变量
  - name: environment
    type: enum
    options: ["development", "staging", "production"]
    default: "development"
    description: "部署环境"

  # 数组变量
  - name: features
    type: array
    item_type: string
    description: "功能列表"

  # 对象变量
  - name: database
    type: object
    schema:
      host: { type: string, required: true }
      port: { type: number, default: 5432 }
      name: { type: string, required: true }

  # 计算变量
  - name: full_name
    type: computed
    expression: "${first_name} ${last_name}"
    description: "完整姓名"
```

### 变量引用
```markdown
## 基本引用
${variable_name}                    # 简单变量引用
${object.property}                  # 对象属性引用
${array[0]}                        # 数组元素引用

## 高级引用
${variable_name|default:"默认值"}    # 默认值
${variable_name|upper}              # 转大写
${variable_name|lower}              # 转小写
${variable_name|capitalize}         # 首字母大写
${variable_name|length}             # 长度
${date|format:"YYYY-MM-DD"}         # 日期格式化

## 条件引用
${condition ? value_if_true : value_if_false}
${variable_name|exists ? "存在" : "不存在"}
```

## 4. 控制结构

### 条件语句
```markdown
## 基本条件
{% if condition %}
  内容在条件为真时显示
{% endif %}

{% if condition %}
  条件为真的内容
{% else %}
  条件为假的内容
{% endif %}

{% if condition1 %}
  条件1为真
{% elif condition2 %}
  条件2为真
{% else %}
  所有条件都为假
{% endif %}

## 条件表达式
- ${variable} == "value"
- ${number} > 10
- ${array|length} > 0
- ${variable|exists}
- ${string|contains:"substring"}
```

### 循环语句
```markdown
## 数组循环
{% for item in array %}
- ${item}
{% endfor %}

## 对象循环
{% for key, value in object %}
- ${key}: ${value}
{% endfor %}

## 范围循环
{% for i in range(1, 5) %}
${i}. 项目
{% endfor %}

## 循环变量
{% for item in items %}
  ${loop.index}: ${item}     # 索引（从1开始）
  ${loop.index0}: ${item}    # 索引（从0开始）
  ${loop.first}: ${item}     # 是否第一个
  ${loop.last}: ${item}      # 是否最后一个
  ${loop.length}: ${item}    # 总数量
{% endfor %}
```

### 包含语句
```markdown
## 模板包含
{% include "other_template.md" %}

## 带变量包含
{% include "header_template.md" with {title: "页面标题"} %}

## 条件包含
{% if show_footer %}
  {% include "footer_template.md" %}
{% endif %}
```

## 5. 函数和过滤器

### 内置函数
```markdown
## 字符串函数
- upper(string): 转大写
- lower(string): 转小写
- capitalize(string): 首字母大写
- trim(string): 去除首尾空白
- replace(string, old, new): 字符串替换
- substring(string, start, length): 子字符串

## 数组函数
- length(array): 数组长度
- join(array, separator): 数组连接
- sort(array): 数组排序
- unique(array): 去重
- filter(array, condition): 过滤
- map(array, expression): 映射

## 日期函数
- now(): 当前时间
- format_date(date, format): 日期格式化
- add_days(date, days): 增加天数
- diff_days(date1, date2): 日期差

## 工具函数
- random(min, max): 随机数
- uuid(): 生成UUID
- hash(string): 计算哈希
- encode_base64(string): Base64编码
```

### 自定义函数
```yaml
functions:
  - name: generate_password
    description: "生成随机密码"
    parameters:
      - name: length
        type: number
        default: 12
    implementation: |
      import random
      import string
      chars = string.ascii_letters + string.digits + "!@#$%"
      return ''.join(random.choice(chars) for _ in range(length))
```

## 6. 模板继承

### 基础模板
```markdown
<!-- base_template.md -->
---
template:
  name: "基础模板"
  type: "base"
---

# {% block title %}默认标题{% endblock %}

## 概述
{% block overview %}
这是默认概述内容
{% endblock %}

## 详细信息
{% block content %}
<!-- 子模板内容将在此处显示 -->
{% endblock %}

## 结论
{% block conclusion %}
{% endblock %}
```

### 子模板
```markdown
<!-- specific_template.md -->
---
template:
  extends: "base_template.md"
  name: "特定模板"
---

{% block title %}
特定项目标题
{% endblock %}

{% block overview %}
这是特定的概述内容，覆盖了基础模板的默认内容。
{% endblock %}

{% block content %}
## 特定内容

这是特定模板的详细内容。

### 子章节
- 项目1
- 项目2
- 项目3
{% endblock %}

{% block conclusion %}
## 总结

这是特定的结论部分。
{% endblock %}
```

## 7. 模板组合

### 组件模板
```markdown
<!-- components/header.md -->
---
template:
  type: "component"
  name: "页面头部"
variables:
  - name: title
    type: string
    required: true
  - name: subtitle
    type: string
    required: false
---

# ${title}
{% if subtitle %}
## ${subtitle}
{% endif %}

**日期**: ${now()|format_date:"YYYY-MM-DD"}
**版本**: ${version}
```

### 使用组件
```markdown
<!-- main_template.md -->
---
template:
  name: "主模板"
variables:
  - name: project_title
    type: string
  - name: project_version
    type: string
---

{% include "components/header.md" with {
  title: "${project_title}",
  subtitle: "项目文档",
  version: "${project_version}"
} %}

## 正文内容
这里是主要内容...

{% include "components/footer.md" %}
```

## 8. 验证和测试

### 模板验证
```yaml
validation:
  syntax_check: true           # 语法检查
  variable_check: true         # 变量检查
  reference_check: true        # 引用检查
  output_validation: true      # 输出验证

rules:
  - no_undefined_variables: true
  - required_variables_present: true
  - valid_template_references: true
  - proper_control_structure: true
```

### 测试用例
```yaml
tests:
  - name: "基础功能测试"
    input:
      project_name: "测试项目"
      version: "1.0.0"
    expected_output: |
      # 测试项目
      版本: 1.0.0

  - name: "条件逻辑测试"
    input:
      show_section: true
      content: "测试内容"
    expected_contains:
      - "测试内容"

  - name: "循环功能测试"
    input:
      items: ["项目1", "项目2", "项目3"]
    expected_pattern: "- 项目\\d"
```

## 9. YAML模板系统

### 9.1 概述

除了传统的Markdown模板系统外，我们还提供了基于YAML的模板定义系统，具有更强的逻辑性和结构化特性：

**优势对比**:
```markdown
## 传统Markdown模板
✅ 直观易读
✅ 简单变量替换
❌ 逻辑分散在文本中
❌ 缺乏类型验证
❌ 复杂逻辑难以维护

## YAML模板系统
✅ 结构化逻辑定义
✅ 强类型变量系统
✅ 模块化组件设计
✅ 更好的验证机制
✅ 清晰的控制流程
❌ 学习成本稍高
```

### 9.2 使用场景选择

```yaml
template_selection:
  markdown_templates:
    适用场景:
      - 简单文档生成
      - 快速原型开发
      - 内容为主的模板
      - 学习和入门
    示例:
      - 简单的README
      - 基础文档模板
      - 快速笔记模板

  yaml_templates:
    适用场景:
      - 复杂逻辑控制
      - 多条件分支
      - 大型文档系统
      - 企业级应用
    示例:
      - 技术规格文档
      - API文档生成
      - 测试报告
      - 配置文件生成
```

### 9.3 基本结构

```yaml
# template.yaml
metadata:
  name: "模板名称"
  version: "1.0.0"
  description: "模板描述"

variables:
  # 强类型变量定义
  project_name:
    type: string
    required: true
    validation:
      pattern: "^[a-zA-Z][a-zA-Z0-9_-]*$"

structure:
  # 结构化模板定义
  type: linear
  sections:
    - id: header
      type: component_ref
      component: header_component

logic:
  # 逻辑控制
  conditions:
    - name: has_features
      expression: "${features|length} > 0"

components:
  # 可重用组件
  header_component:
    structure:
      sections:
        - type: content
          content: "# ${project_name}"

output:
  format: markdown
```

### 9.4 核心特性

#### 9.4.1 强类型变量系统
```yaml
variables:
  # 基础类型
  name: { type: string, required: true }
  count: { type: number, min: 1, max: 100 }
  enabled: { type: boolean, default: false }

  # 复合类型
  features:
    type: array
    item_type: string
    min_items: 1

  config:
    type: object
    schema:
      host: { type: string, required: true }
      port: { type: number, default: 8080 }

  # 计算变量
  full_title:
    type: computed
    expression: "${name} v${version}"
    dependencies: ["name", "version"]
```

#### 9.4.2 结构化逻辑控制
```yaml
logic:
  conditions:
    - name: is_production
      expression: "${env} == 'production'"

  rules:
    - if: is_production
      then:
        - remove_section: debug_info
        - set_variable:
            log_level: "ERROR"
      else:
        - set_variable:
            log_level: "DEBUG"
```

#### 9.4.3 模块化组件系统
```yaml
components:
  api_endpoint:
    parameters:
      - endpoint: { type: object, required: true }
    structure:
      sections:
        - type: content
          content: "### ${endpoint.method} ${endpoint.path}"
        - type: conditional
          condition: "${endpoint.description|exists}"
          structure:
            sections:
              - type: content
                content: "${endpoint.description}"
```

### 9.5 详细规范

关于YAML模板系统的完整规范和示例，请参考：
- [YAML模板详细规范](yaml-template-specification.md)
- [YAML模板示例集](yaml-template-examples.md)

### 9.6 工具支持

#### 9.6.1 模板转换工具
```bash
# Markdown模板转YAML模板
vibe-template convert --from markdown --to yaml input.md output.yaml

# YAML模板转Markdown模板
vibe-template convert --from yaml --to markdown input.yaml output.md

# 验证YAML模板
vibe-template validate template.yaml
```

#### 9.6.2 混合使用
```yaml
project_templates:
  markdown:
    - simple_readme.md        # 简单文档
    - quick_notes.md          # 快速笔记
  yaml:
    - tech_spec.yaml          # 技术规格
    - api_docs.yaml           # API文档
    - test_report.yaml        # 测试报告

template_engine:
  auto_detect: true           # 自动检测模板格式
  fallback: markdown          # 默认格式
```