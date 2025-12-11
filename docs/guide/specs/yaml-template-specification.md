# YAML 模板规范

## 1. 概述

YAML模板系统提供了更强的逻辑性和结构化的模板定义方式，相比传统的Markdown+变量替换方式，YAML模板具有以下优势：

- **结构化逻辑**: 所有逻辑都在YAML结构中定义，便于理解和维护
- **类型安全**: 强类型变量定义和验证
- **模块化**: 更好的模板组合和重用能力
- **可视化**: 逻辑流程更加清晰可读

## 2. 基本结构

### 2.1 文件格式
```yaml
# {template-name}.yaml
metadata:
  name: "模板名称"
  version: "1.0.0"
  description: "模板描述"
  category: "模板分类"
  author: "作者信息"
  created: "2024-01-01"
  updated: "2024-01-01"
  tags: ["tag1", "tag2"]

variables:
  # 变量定义

structure:
  # 模板结构定义

logic:
  # 逻辑控制

functions:
  # 自定义函数

output:
  # 输出配置
```

### 2.2 核心组件

## 3. 变量系统

### 3.1 变量定义
```yaml
variables:
  # 简单变量
  project_name:
    type: string
    required: true
    default: ""
    description: "项目名称"
    validation:
      pattern: "^[a-zA-Z][a-zA-Z0-9_-]*$"

  # 枚举变量
  environment:
    type: enum
    options: ["development", "staging", "production"]
    default: "development"
    description: "部署环境"

  # 复合变量
  database:
    type: object
    schema:
      host:
        type: string
        required: true
      port:
        type: number
        default: 5432
        validation:
          min: 1
          max: 65535
      name:
        type: string
        required: true

  # 数组变量
  features:
    type: array
    item_type: string
    min_items: 1
    description: "功能列表"

  # 计算变量
  full_name:
    type: computed
    expression: "${first_name} ${last_name}"
    dependencies: ["first_name", "last_name"]
```

### 3.2 变量类型
```yaml
# 支持的变量类型
types:
  primitive:
    - string
    - number
    - boolean
    - date
    - url
    - email
    - file_path

  composite:
    - array
    - object
    - enum

  special:
    - computed
    - template_ref
    - function_ref
```

## 4. 模板结构

### 4.1 线性结构
```yaml
structure:
  type: linear
  sections:
    - id: header
      type: component
      template: header_component
      variables:
        title: "${project_name}"

    - id: overview
      type: content
      content: |
        ## 项目概述
        ${project_description}

    - id: features
      type: loop
      iterate: features
      template: feature_item

    - id: footer
      type: component
      template: footer_component
```

### 4.2 条件结构
```yaml
structure:
  type: conditional
  conditions:
    - condition: "${environment} == 'production'"
      structure:
        - type: content
          content: "## 生产环境配置"
    - condition: "${environment} == 'development'"
      structure:
        - type: content
          content: "## 开发环境配置"
    - default:
        - type: content
          content: "## 默认配置"
```

### 4.3 分支结构
```yaml
structure:
  type: branched
  branches:
    api_doc:
      condition: "${doc_type} == 'api'"
      sections:
        - type: component
          template: api_header
        - type: loop
          iterate: endpoints
          template: endpoint_doc

    user_guide:
      condition: "${doc_type} == 'guide'"
      sections:
        - type: component
          template: guide_header
        - type: content
          content: "${guide_content}"
```

## 5. 逻辑控制

### 5.1 条件逻辑
```yaml
logic:
  conditions:
    - name: has_database
      expression: "${database.host|exists}"

    - name: is_production
      expression: "${environment} == 'production'"

    - name: has_features
      expression: "${features|length} > 0"

  rules:
    - if: has_database
      then:
        - add_section: database_config
        - set_variable:
            db_url: "postgresql://${database.host}:${database.port}/${database.name}"

    - if: is_production
      then:
        - remove_section: debug_info
        - set_variable:
            log_level: "ERROR"
      else:
        - set_variable:
            log_level: "DEBUG"
```

### 5.2 循环逻辑
```yaml
logic:
  loops:
    - name: feature_loop
      iterate: features
      item_name: feature
      index_name: idx
      actions:
        - generate_section:
            template: feature_detail
            variables:
              feature_name: "${feature.name}"
              feature_index: "${idx + 1}"

    - name: env_loop
      iterate: environments
      filter: "${item.enabled}"
      sort_by: "priority"
      actions:
        - append_content: "- 环境: ${item.name}"
```

### 5.3 数据变换
```yaml
logic:
  transforms:
    - name: format_features
      input: features
      operations:
        - filter: "${item.enabled}"
        - map: "${item.name|upper}"
        - sort: "asc"
      output: formatted_features

    - name: group_endpoints
      input: api_endpoints
      group_by: "category"
      output: grouped_endpoints
```

## 6. 函数系统

### 6.1 内置函数
```yaml
functions:
  builtin:
    string:
      - upper(str) -> str
      - lower(str) -> str
      - capitalize(str) -> str
      - trim(str) -> str
      - replace(str, old, new) -> str
      - split(str, delimiter) -> array

    array:
      - length(array) -> number
      - join(array, separator) -> str
      - filter(array, condition) -> array
      - map(array, expression) -> array
      - sort(array, order) -> array

    object:
      - keys(object) -> array
      - values(object) -> array
      - merge(obj1, obj2) -> object

    date:
      - now() -> date
      - format_date(date, format) -> str
      - add_days(date, days) -> date
      - diff_days(date1, date2) -> number
```

### 6.2 自定义函数
```yaml
functions:
  custom:
    - name: generate_uuid
      description: "生成UUID"
      return_type: string
      implementation:
        type: javascript
        code: |
          function() {
            return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
              var r = Math.random() * 16 | 0;
              var v = c == 'x' ? r : (r & 0x3 | 0x8);
              return v.toString(16);
            });
          }

    - name: calculate_complexity
      description: "计算复杂度分数"
      parameters:
        - name: features
          type: array
        - name: weight
          type: number
          default: 1.0
      return_type: number
      implementation:
        type: python
        code: |
          def calculate_complexity(features, weight=1.0):
              base_score = len(features) * 10
              complexity_bonus = sum(f.get('complexity', 1) for f in features)
              return int((base_score + complexity_bonus) * weight)
```

## 7. 模板组合

### 7.1 组件引用
```yaml
components:
  header:
    file: components/header.yaml
    variables:
      title: "${project_name}"
      subtitle: "${project_subtitle}"

  footer:
    file: components/footer.yaml
    variables:
      version: "${version}"
      generated_at: "${now()|format_date:'YYYY-MM-DD HH:mm:ss'}"

structure:
  sections:
    - type: component_ref
      component: header
    - type: content
      content: "主要内容"
    - type: component_ref
      component: footer
```

### 7.2 模板继承
```yaml
# base_template.yaml
metadata:
  name: "基础模板"
  type: base

blocks:
  title:
    default: "默认标题"
    type: string

  content:
    required: true
    type: structure

  footer:
    default: ""
    type: string

structure:
  sections:
    - type: content
      content: "# ${blocks.title}"
    - type: block_ref
      block: content
    - type: content
      content: "${blocks.footer}"

---

# child_template.yaml
metadata:
  name: "子模板"
  extends: base_template.yaml

blocks:
  title: "特定标题"
  content:
    sections:
      - type: content
        content: "## 子模板内容"
      - type: loop
        iterate: items
        template: item_template
  footer: "子模板页脚"
```

## 8. 输出配置

### 8.1 输出格式
```yaml
output:
  format: markdown
  options:
    line_ending: "\n"
    indentation: "  "
    encoding: "utf-8"

  post_process:
    - trim_empty_lines
    - normalize_whitespace
    - validate_markdown

  validation:
    - check_syntax
    - validate_links
    - check_formatting
```

### 8.2 多格式输出
```yaml
output:
  formats:
    markdown:
      file_extension: ".md"
      processor: markdown_processor
      options:
        toc: true
        code_highlight: true

    html:
      file_extension: ".html"
      processor: html_processor
      template: html_wrapper.html

    pdf:
      file_extension: ".pdf"
      processor: pdf_processor
      options:
        page_size: "A4"
        margins: "2cm"
```

## 9. 验证和测试

### 9.1 模板验证
```yaml
validation:
  schema:
    strict_mode: true
    check_required_variables: true
    validate_expressions: true
    check_template_references: true

  rules:
    - no_undefined_variables
    - valid_function_calls
    - proper_loop_structure
    - consistent_types

  custom_validators:
    - name: check_project_name
      expression: "project_name|matches:'^[a-zA-Z][\\w-]*$'"
      message: "项目名称必须以字母开头，只能包含字母、数字、下划线和短横线"
```

### 9.2 测试用例
```yaml
tests:
  - name: "基础功能测试"
    description: "测试基本变量替换"
    input:
      project_name: "test-project"
      version: "1.0.0"
    expected:
      contains:
        - "# test-project"
        - "版本: 1.0.0"
      not_contains:
        - "${project_name}"

  - name: "条件逻辑测试"
    description: "测试条件渲染"
    input:
      environment: "production"
      debug_mode: false
    expected:
      contains:
        - "生产环境配置"
      not_contains:
        - "调试信息"

  - name: "循环功能测试"
    description: "测试循环渲染"
    input:
      features:
        - name: "用户认证"
          enabled: true
        - name: "数据分析"
          enabled: false
    expected:
      pattern: "- 用户认证"
      not_pattern: "- 数据分析"
```

## 10. 工具支持

### 10.1 模板编译器
```yaml
compiler:
  input: template.yaml
  output:
    - template.md
    - template.html
    - template.json

  optimization:
    - dead_code_elimination
    - constant_folding
    - template_inlining

  debug:
    - source_maps
    - variable_tracking
    - execution_trace
```

### 10.2 IDE支持
```yaml
ide_support:
  language_server:
    - syntax_highlighting
    - auto_completion
    - error_detection
    - goto_definition
    - find_references

  extensions:
    - vscode: vibe-yaml-templates
    - intellij: yaml-template-plugin
    - vim: yaml-template.vim
```

## 11. 示例对比

### 11.1 传统Markdown模板
```markdown
---
template:
  name: "API文档"
variables:
  - name: api_name
    type: string
  - name: endpoints
    type: array
---

# ${api_name} API文档

## 端点列表

{% for endpoint in endpoints %}
### ${endpoint.method} ${endpoint.path}

**描述**: ${endpoint.description}

{% if endpoint.parameters %}
**参数**:
{% for param in endpoint.parameters %}
- `${param.name}` (${param.type}): ${param.description}
{% endfor %}
{% endif %}

{% endfor %}
```

### 11.2 YAML模板
```yaml
metadata:
  name: "API文档模板"
  version: "2.0.0"

variables:
  api_name:
    type: string
    required: true
    description: "API名称"

  endpoints:
    type: array
    item_type: object
    schema:
      method: { type: enum, options: ["GET", "POST", "PUT", "DELETE"] }
      path: { type: string, required: true }
      description: { type: string, required: true }
      parameters:
        type: array
        item_type: object
        schema:
          name: { type: string, required: true }
          type: { type: string, required: true }
          description: { type: string, required: true }

structure:
  type: linear
  sections:
    - type: content
      content: "# ${api_name} API文档"

    - type: content
      content: "## 端点列表"

    - type: loop
      iterate: endpoints
      item_name: endpoint
      template: endpoint_section

components:
  endpoint_section:
    structure:
      sections:
        - type: content
          content: "### ${endpoint.method} ${endpoint.path}"

        - type: content
          content: "**描述**: ${endpoint.description}"

        - type: conditional
          condition: "${endpoint.parameters|length} > 0"
          then:
            - type: content
              content: "**参数**:"
            - type: loop
              iterate: endpoint.parameters
              item_name: param
              template: parameter_item

  parameter_item:
    structure:
      sections:
        - type: content
          content: "- `${param.name}` (${param.type}): ${param.description}"

logic:
  validation:
    - name: valid_endpoints
      expression: "${endpoints|length} > 0"
      message: "至少需要定义一个端点"

output:
  format: markdown
  options:
    validate_structure: true
```

## 12. 迁移指南

### 12.1 从Markdown模板迁移
1. **提取元数据**: 将YAML front matter转换为metadata部分
2. **重构变量**: 将变量定义转换为强类型定义
3. **分离逻辑**: 将模板中的逻辑提取到logic部分
4. **结构化内容**: 将线性内容转换为结构化定义

### 12.2 迁移工具
```yaml
migration_tools:
  converter:
    name: "md2yaml"
    usage: "md2yaml input.md output.yaml"
    features:
      - automatic_extraction
      - logic_detection
      - variable_inference

  validator:
    name: "yaml-template-lint"
    usage: "yaml-template-lint template.yaml"
    checks:
      - syntax_validation
      - type_checking
      - reference_validation
```

这个YAML模板系统提供了更强的逻辑性、更好的类型安全和更清晰的结构，同时保持了灵活性和可扩展性。
