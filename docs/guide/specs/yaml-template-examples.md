# YAML 模板示例集

## 1. 技术规格模板示例

### 1.1 TECH_SPEC模板
```yaml
# tech_spec_template.yaml
metadata:
  name: "技术规格文档模板"
  version: "2.0.0"
  description: "用于生成标准化技术规格文档"
  category: "documentation"
  author: "vibe-coding"
  tags: ["tech-spec", "documentation", "engineering"]

variables:
  # 基础信息
  project_name:
    type: string
    required: true
    description: "项目名称"
    validation:
      pattern: "^[a-zA-Z][a-zA-Z0-9_-]*$"

  project_version:
    type: string
    required: true
    description: "项目版本"
    validation:
      pattern: "^\\d+\\.\\d+\\.\\d+$"

  author:
    type: string
    required: true
    description: "文档作者"

  # 项目配置
  tech_stack:
    type: object
    required: true
    schema:
      backend:
        type: array
        item_type: string
        description: "后端技术栈"
      frontend:
        type: array
        item_type: string
        description: "前端技术栈"
      database:
        type: string
        description: "数据库类型"
      deployment:
        type: string
        description: "部署方式"

  # 功能模块
  modules:
    type: array
    required: true
    item_type: object
    schema:
      name:
        type: string
        required: true
      description:
        type: string
        required: true
      priority:
        type: enum
        options: ["高", "中", "低"]
        default: "中"
      complexity:
        type: number
        min: 1
        max: 10
        default: 5
      dependencies:
        type: array
        item_type: string
        description: "依赖的其他模块"

  # 配置选项
  include_architecture:
    type: boolean
    default: true
    description: "是否包含架构设计部分"

  include_api_design:
    type: boolean
    default: true
    description: "是否包含API设计部分"

  include_database_design:
    type: boolean
    default: true
    description: "是否包含数据库设计部分"

# 计算变量
computed_variables:
  total_modules:
    expression: "${modules|length}"
    description: "模块总数"

  high_priority_modules:
    expression: "${modules|filter:'priority == \"高\"'|length}"
    description: "高优先级模块数量"

  avg_complexity:
    expression: "${modules|map:'complexity'|avg|round:1}"
    description: "平均复杂度"

structure:
  type: linear
  sections:
    - id: header
      type: component_ref
      component: document_header

    - id: overview
      type: content
      content: |
        ## 1. 项目概述

        **项目名称**: ${project_name}
        **版本**: ${project_version}
        **作者**: ${author}
        **创建时间**: ${now()|format_date:'YYYY-MM-DD'}

        ### 1.1 技术栈

        **后端技术**: ${tech_stack.backend|join:', '}
        **前端技术**: ${tech_stack.frontend|join:', '}
        **数据库**: ${tech_stack.database}
        **部署方式**: ${tech_stack.deployment}

        ### 1.2 模块统计

        - 总模块数: ${total_modules}
        - 高优先级模块: ${high_priority_modules}
        - 平均复杂度: ${avg_complexity}/10

    - id: architecture
      type: conditional
      condition: "${include_architecture}"
      structure:
        sections:
          - type: component_ref
            component: architecture_section

    - id: modules
      type: content
      content: "## 3. 功能模块"

    - id: module_list
      type: loop
      iterate: modules
      item_name: module
      index_name: module_idx
      template: module_detail

    - id: api_design
      type: conditional
      condition: "${include_api_design}"
      structure:
        sections:
          - type: component_ref
            component: api_design_section

    - id: database_design
      type: conditional
      condition: "${include_database_design}"
      structure:
        sections:
          - type: component_ref
            component: database_design_section

components:
  document_header:
    structure:
      sections:
        - type: content
          content: |
            # ${project_name} 技术规格文档

            **文档版本**: ${project_version}
            **最后更新**: ${now()|format_date:'YYYY-MM-DD HH:mm:ss'}
            **状态**: 草稿

  architecture_section:
    structure:
      sections:
        - type: content
          content: |
            ## 2. 系统架构

            ### 2.1 整体架构

            > TODO: 添加系统架构图

            ### 2.2 技术选型

            | 层级 | 技术选择 | 说明 |
            |------|----------|------|
            | 后端 | ${tech_stack.backend|join:', '} | 服务端技术栈 |
            | 前端 | ${tech_stack.frontend|join:', '} | 客户端技术栈 |
            | 数据库 | ${tech_stack.database} | 数据存储方案 |
            | 部署 | ${tech_stack.deployment} | 部署策略 |

  module_detail:
    structure:
      sections:
        - type: content
          content: |
            ### 3.${module_idx + 1} ${module.name}

            **描述**: ${module.description}
            **优先级**: ${module.priority}
            **复杂度**: ${module.complexity}/10
        - type: conditional
          condition: "${module.dependencies|length} > 0"
          then:
            sections:
              - type: content
                content: |
                  **依赖模块**: ${module.dependencies|join:', '}

  api_design_section:
    structure:
      sections:
        - type: content
          content: |
            ## 4. API设计

            ### 4.1 接口规范

            - **协议**: RESTful API
            - **数据格式**: JSON
            - **认证方式**: JWT Token

            ### 4.2 接口列表

            > TODO: 添加具体的API接口定义

  database_design_section:
    structure:
      sections:
        - type: content
          content: |
            ## 5. 数据库设计

            ### 5.1 数据库类型

            **数据库**: ${tech_stack.database}

            ### 5.2 表结构设计

            > TODO: 添加数据库表结构设计

logic:
  validation:
    - name: has_modules
      expression: "${modules|length} > 0"
      message: "至少需要定义一个功能模块"

    - name: valid_tech_stack
      expression: "${tech_stack.backend|length} > 0 && ${tech_stack.frontend|length} > 0"
      message: "必须定义前端和后端技术栈"

output:
  format: markdown
  options:
    toc: true
    line_ending: "\n"
    validate_structure: true
  post_process:
    - normalize_whitespace
    - validate_markdown
```

## 2. API文档模板示例

### 2.1 API文档模板
```yaml
# api_docs_template.yaml
metadata:
  name: "API文档模板"
  version: "2.0.0"
  description: "用于生成RESTful API文档"
  category: "api-documentation"

variables:
  api_info:
    type: object
    required: true
    schema:
      name:
        type: string
        required: true
      version:
        type: string
        required: true
      description:
        type: string
        required: true
      base_url:
        type: url
        required: true

  authentication:
    type: object
    schema:
      type:
        type: enum
        options: ["JWT", "API_KEY", "OAUTH2", "BASIC"]
        required: true
      description:
        type: string
        required: true
      header_name:
        type: string
        default: "Authorization"

  endpoints:
    type: array
    required: true
    min_items: 1
    item_type: object
    schema:
      method:
        type: enum
        options: ["GET", "POST", "PUT", "DELETE", "PATCH"]
        required: true
      path:
        type: string
        required: true
      summary:
        type: string
        required: true
      description:
        type: string
      tags:
        type: array
        item_type: string
      parameters:
        type: array
        item_type: object
        schema:
          name: { type: string, required: true }
          type: { type: string, required: true }
          location: { type: enum, options: ["query", "path", "header", "body"] }
          required: { type: boolean, default: false }
          description: { type: string }
      responses:
        type: array
        item_type: object
        schema:
          status_code: { type: number, required: true }
          description: { type: string, required: true }
          example: { type: string }

computed_variables:
  endpoints_by_tag:
    expression: "${endpoints|group_by:'tags[0]'}"
    description: "按标签分组的端点"

  total_endpoints:
    expression: "${endpoints|length}"
    description: "端点总数"

structure:
  type: linear
  sections:
    - id: header
      type: content
      content: |
        # ${api_info.name} API文档

        **版本**: ${api_info.version}
        **基础URL**: ${api_info.base_url}
        **更新时间**: ${now()|format_date:'YYYY-MM-DD HH:mm:ss'}

        ## 概述

        ${api_info.description}

        **端点总数**: ${total_endpoints}

    - id: authentication
      type: content
      content: |
        ## 认证方式

        **认证类型**: ${authentication.type}
        **请求头**: ${authentication.header_name}

        ${authentication.description}

    - id: endpoints_by_category
      type: loop
      iterate: endpoints_by_tag
      item_name: category
      template: category_section

components:
  category_section:
    structure:
      sections:
        - type: content
          content: "## ${category.key|default:'未分类'} 接口"

        - type: loop
          iterate: category.value
          item_name: endpoint
          template: endpoint_detail

  endpoint_detail:
    structure:
      sections:
        - type: content
          content: |
            ### ${endpoint.method} ${endpoint.path}

            **摘要**: ${endpoint.summary}

        - type: conditional
          condition: "${endpoint.description|exists}"
          then:
            sections:
              - type: content
                content: |
                  **描述**: ${endpoint.description}

        - type: conditional
          condition: "${endpoint.parameters|length} > 0"
          then:
            sections:
              - type: content
                content: "#### 请求参数"
              - type: loop
                iterate: endpoint.parameters
                item_name: param
                template: parameter_detail

        - type: conditional
          condition: "${endpoint.responses|length} > 0"
          then:
            sections:
              - type: content
                content: "#### 响应"
              - type: loop
                iterate: endpoint.responses
                item_name: response
                template: response_detail

  parameter_detail:
    structure:
      sections:
        - type: content
          content: |
            | 参数名 | 类型 | 位置 | 必需 | 描述 |
            |--------|------|------|------|------|
            | `${param.name}` | ${param.type} | ${param.location} | ${param.required ? '是' : '否'} | ${param.description|default:'-'} |

  response_detail:
    structure:
      sections:
        - type: content
          content: |
            **状态码 ${response.status_code}**: ${response.description}

        - type: conditional
          condition: "${response.example|exists}"
          then:
            sections:
              - type: content
                content: |
                  ```json
                  ${response.example}
                  ```

output:
  format: markdown
  options:
    validate_structure: true
    toc: true
```

## 3. 测试报告模板示例

### 3.1 测试报告模板
```yaml
# test_report_template.yaml
metadata:
  name: "测试报告模板"
  version: "2.0.0"
  description: "用于生成标准化测试报告"
  category: "testing"

variables:
  test_info:
    type: object
    required: true
    schema:
      project_name: { type: string, required: true }
      version: { type: string, required: true }
      test_type: { type: enum, options: ["单元测试", "集成测试", "端到端测试", "性能测试"] }
      test_date: { type: date, required: true }
      tester: { type: string, required: true }

  test_environment:
    type: object
    schema:
      os: { type: string }
      browser: { type: string }
      device: { type: string }
      test_framework: { type: string }

  test_suites:
    type: array
    required: true
    item_type: object
    schema:
      name: { type: string, required: true }
      description: { type: string }
      total_tests: { type: number, required: true }
      passed_tests: { type: number, required: true }
      failed_tests: { type: number, required: true }
      skipped_tests: { type: number, default: 0 }
      execution_time: { type: number, description: "执行时间（秒）" }
      test_cases:
        type: array
        item_type: object
        schema:
          name: { type: string, required: true }
          status: { type: enum, options: ["通过", "失败", "跳过"] }
          duration: { type: number }
          error_message: { type: string }

  performance_metrics:
    type: object
    schema:
      response_time: { type: number }
      throughput: { type: number }
      memory_usage: { type: number }
      cpu_usage: { type: number }

computed_variables:
  total_tests:
    expression: "${test_suites|map:'total_tests'|sum}"

  total_passed:
    expression: "${test_suites|map:'passed_tests'|sum}"

  total_failed:
    expression: "${test_suites|map:'failed_tests'|sum}"

  total_skipped:
    expression: "${test_suites|map:'skipped_tests'|sum}"

  pass_rate:
    expression: "${(total_passed / total_tests * 100)|round:2}"

  total_execution_time:
    expression: "${test_suites|map:'execution_time'|sum}"

structure:
  type: linear
  sections:
    - id: header
      type: component_ref
      component: report_header

    - id: summary
      type: component_ref
      component: test_summary

    - id: environment
      type: component_ref
      component: environment_info

    - id: test_suites
      type: content
      content: "## 测试详情"

    - id: suite_details
      type: loop
      iterate: test_suites
      item_name: suite
      index_name: suite_idx
      template: suite_detail

    - id: performance
      type: conditional
      condition: "${performance_metrics|exists}"
      structure:
        sections:
          - type: component_ref
            component: performance_section

    - id: conclusion
      type: component_ref
      component: test_conclusion

components:
  report_header:
    structure:
      sections:
        - type: content
          content: |
            # ${test_info.project_name} ${test_info.test_type}报告

            **项目版本**: ${test_info.version}
            **测试日期**: ${test_info.test_date|format_date:'YYYY-MM-DD'}
            **测试人员**: ${test_info.tester}
            **报告生成时间**: ${now()|format_date:'YYYY-MM-DD HH:mm:ss'}

  test_summary:
    structure:
      sections:
        - type: content
          content: |
            ## 测试概览

            | 指标 | 数值 |
            |------|------|
            | 总测试数 | ${total_tests} |
            | 通过数 | ${total_passed} |
            | 失败数 | ${total_failed} |
            | 跳过数 | ${total_skipped} |
            | 通过率 | ${pass_rate}% |
            | 总执行时间 | ${total_execution_time}秒 |

            ### 测试结果概要

        - type: conditional
          condition: "${total_failed} == 0"
          then:
            sections:
              - type: content
                content: "✅ **所有测试通过！**"
          else:
            sections:
              - type: content
                content: "❌ **存在 ${total_failed} 个失败的测试用例**"

  environment_info:
    structure:
      sections:
        - type: content
          content: |
            ## 测试环境

            | 环境项 | 信息 |
            |--------|------|
        - type: conditional
          condition: "${test_environment.os|exists}"
          then:
            sections:
              - type: content
                content: "| 操作系统 | ${test_environment.os} |"
        - type: conditional
          condition: "${test_environment.browser|exists}"
          then:
            sections:
              - type: content
                content: "| 浏览器 | ${test_environment.browser} |"
        - type: conditional
          condition: "${test_environment.device|exists}"
          then:
            sections:
              - type: content
                content: "| 设备 | ${test_environment.device} |"
        - type: conditional
          condition: "${test_environment.test_framework|exists}"
          then:
            sections:
              - type: content
                content: "| 测试框架 | ${test_environment.test_framework} |"

  suite_detail:
    structure:
      sections:
        - type: content
          content: |
            ### ${suite_idx + 1}. ${suite.name}

        - type: conditional
          condition: "${suite.description|exists}"
          then:
            sections:
              - type: content
                content: "${suite.description}"

        - type: content
          content: |
            **测试统计**:
            - 总数: ${suite.total_tests}
            - 通过: ${suite.passed_tests}
            - 失败: ${suite.failed_tests}
            - 跳过: ${suite.skipped_tests}
            - 执行时间: ${suite.execution_time}秒

        - type: conditional
          condition: "${suite.test_cases|length} > 0"
          then:
            sections:
              - type: content
                content: "#### 测试用例详情"
              - type: loop
                iterate: suite.test_cases
                item_name: test_case
                template: test_case_detail

  test_case_detail:
    structure:
      sections:
        - type: content
          content: |
            - **${test_case.name}**: ${test_case.status}
        - type: conditional
          condition: "${test_case.duration|exists}"
          then:
            sections:
              - type: content
                content: " (${test_case.duration}秒)"
        - type: conditional
          condition: "${test_case.status} == '失败' && ${test_case.error_message|exists}"
          then:
            sections:
              - type: content
                content: |

                  ```
                  错误信息: ${test_case.error_message}
                  ```

  performance_section:
    structure:
      sections:
        - type: content
          content: |
            ## 性能指标

            | 指标 | 数值 |
            |------|------|
        - type: conditional
          condition: "${performance_metrics.response_time|exists}"
          then:
            sections:
              - type: content
                content: "| 平均响应时间 | ${performance_metrics.response_time}ms |"
        - type: conditional
          condition: "${performance_metrics.throughput|exists}"
          then:
            sections:
              - type: content
                content: "| 吞吐量 | ${performance_metrics.throughput} req/s |"
        - type: conditional
          condition: "${performance_metrics.memory_usage|exists}"
          then:
            sections:
              - type: content
                content: "| 内存使用 | ${performance_metrics.memory_usage}MB |"
        - type: conditional
          condition: "${performance_metrics.cpu_usage|exists}"
          then:
            sections:
              - type: content
                content: "| CPU使用率 | ${performance_metrics.cpu_usage}% |"

  test_conclusion:
    structure:
      sections:
        - type: content
          content: "## 测试结论"
        - type: conditional
          condition: "${pass_rate} >= 95"
          then:
            sections:
              - type: content
                content: |
                  ✅ **测试结果良好**

                  - 通过率达到 ${pass_rate}%，满足发布要求
                  - 建议进行下一阶段测试或发布
          else:
            sections:
              - type: content
                content: |
                  ❌ **测试结果需要改进**

                  - 通过率仅为 ${pass_rate}%，低于95%的目标
                  - 建议修复失败的测试用例后重新测试

        - type: conditional
          condition: "${total_failed} > 0"
          then:
            sections:
              - type: content
                content: |

                  ### 待修复问题

                  需要关注和修复 ${total_failed} 个失败的测试用例。

logic:
  validation:
    - name: valid_test_data
      expression: "${total_tests} > 0"
      message: "必须包含至少一个测试用例"

    - name: consistent_numbers
      expression: "${total_passed + total_failed + total_skipped} == ${total_tests}"
      message: "测试数量统计不一致"

output:
  format: markdown
  options:
    validate_structure: true
    toc: true
```

## 4. 项目README模板示例

### 4.1 README模板
```yaml
# readme_template.yaml
metadata:
  name: "项目README模板"
  version: "2.0.0"
  description: "用于生成标准化的项目README文档"
  category: "documentation"

variables:
  project:
    type: object
    required: true
    schema:
      name: { type: string, required: true }
      description: { type: string, required: true }
      version: { type: string, required: true }
      license: { type: string, default: "MIT" }
      homepage: { type: url }
      repository: { type: url }

  author:
    type: object
    schema:
      name: { type: string }
      email: { type: email }
      url: { type: url }

  tech_stack:
    type: array
    item_type: string
    description: "技术栈"

  features:
    type: array
    item_type: string
    description: "主要功能特性"

  installation:
    type: object
    schema:
      prerequisites:
        type: array
        item_type: string
      steps:
        type: array
        item_type: string

  usage_examples:
    type: array
    item_type: object
    schema:
      title: { type: string, required: true }
      description: { type: string }
      code: { type: string, required: true }
      language: { type: string, default: "bash" }

  contributing:
    type: object
    schema:
      guidelines: { type: string }
      code_of_conduct: { type: string }

  include_badges:
    type: boolean
    default: true

  include_toc:
    type: boolean
    default: true

structure:
  type: linear
  sections:
    - id: header
      type: component_ref
      component: project_header

    - id: badges
      type: conditional
      condition: "${include_badges}"
      structure:
        sections:
          - type: component_ref
            component: badge_section

    - id: toc
      type: conditional
      condition: "${include_toc}"
      structure:
        sections:
          - type: component_ref
            component: table_of_contents

    - id: description
      type: component_ref
      component: description_section

    - id: features
      type: conditional
      condition: "${features|length} > 0"
      structure:
        sections:
          - type: component_ref
            component: features_section

    - id: tech_stack
      type: conditional
      condition: "${tech_stack|length} > 0"
      structure:
        sections:
          - type: component_ref
            component: tech_stack_section

    - id: installation
      type: conditional
      condition: "${installation|exists}"
      structure:
        sections:
          - type: component_ref
            component: installation_section

    - id: usage
      type: conditional
      condition: "${usage_examples|length} > 0"
      structure:
        sections:
          - type: component_ref
            component: usage_section

    - id: contributing
      type: conditional
      condition: "${contributing|exists}"
      structure:
        sections:
          - type: component_ref
            component: contributing_section

    - id: license
      type: component_ref
      component: license_section

components:
  project_header:
    structure:
      sections:
        - type: content
          content: |
            # ${project.name}

            ${project.description}

  badge_section:
    structure:
      sections:
        - type: content
          content: |
            ![Version](https://img.shields.io/badge/version-${project.version}-blue.svg)
            ![License](https://img.shields.io/badge/license-${project.license}-green.svg)

  table_of_contents:
    structure:
      sections:
        - type: content
          content: |
            ## 目录

            - [功能特性](#功能特性)
            - [技术栈](#技术栈)
            - [安装](#安装)
            - [使用方法](#使用方法)
            - [贡献指南](#贡献指南)
            - [许可证](#许可证)

  description_section:
    structure:
      sections:
        - type: content
          content: |
            ## 项目简介

            ${project.description}

            **版本**: ${project.version}

        - type: conditional
          condition: "${project.homepage|exists}"
          then:
            sections:
              - type: content
                content: "**主页**: ${project.homepage}"
        - type: conditional
          condition: "${project.repository|exists}"
          then:
            sections:
              - type: content
                content: "**仓库**: ${project.repository}"

  features_section:
    structure:
      sections:
        - type: content
          content: "## 功能特性"
        - type: loop
          iterate: features
          item_name: feature
          template: feature_item

  feature_item:
    structure:
      sections:
        - type: content
          content: "- ${feature}"

  tech_stack_section:
    structure:
      sections:
        - type: content
          content: |
            ## 技术栈

            ${tech_stack|join:', '}

  installation_section:
    structure:
      sections:
        - type: content
          content: "## 安装"
        - type: conditional
          condition: "${installation.prerequisites|length} > 0"
          then:
            sections:
              - type: content
                content: "### 前置要求"
              - type: loop
                iterate: installation.prerequisites
                item_name: prereq
                template: prerequisite_item
        - type: conditional
          condition: "${installation.steps|length} > 0"
          then:
            sections:
              - type: content
                content: "### 安装步骤"
              - type: loop
                iterate: installation.steps
                item_name: step
                index_name: step_idx
                template: installation_step

  prerequisite_item:
    structure:
      sections:
        - type: content
          content: "- ${prereq}"

  installation_step:
    structure:
      sections:
        - type: content
          content: "${step_idx + 1}. ${step}"

  usage_section:
    structure:
      sections:
        - type: content
          content: "## 使用方法"
        - type: loop
          iterate: usage_examples
          item_name: example
          template: usage_example

  usage_example:
    structure:
      sections:
        - type: content
          content: |
            ### ${example.title}

        - type: conditional
          condition: "${example.description|exists}"
          then:
            sections:
              - type: content
                content: "${example.description}"
        - type: content
          content: |

            ```${example.language}
            ${example.code}
            ```

  contributing_section:
    structure:
      sections:
        - type: content
          content: |
            ## 贡献指南

            欢迎贡献代码！请遵循以下步骤：

            1. Fork 这个仓库
            2. 创建你的功能分支 (`git checkout -b feature/AmazingFeature`)
            3. 提交你的更改 (`git commit -m 'Add some AmazingFeature'`)
            4. 推送到分支 (`git push origin feature/AmazingFeature`)
            5. 打开一个 Pull Request

        - type: conditional
          condition: "${contributing.guidelines|exists}"
          then:
            sections:
              - type: content
                content: |
                  ### 贡献指南

                  ${contributing.guidelines}

        - type: conditional
          condition: "${contributing.code_of_conduct|exists}"
          then:
            sections:
              - type: content
                content: |
                  ### 行为准则

                  ${contributing.code_of_conduct}

  license_section:
    structure:
      sections:
        - type: content
          content: |
            ## 许可证

            本项目基于 ${project.license} 许可证开源。查看 [LICENSE](LICENSE) 文件了解更多详情。

        - type: conditional
          condition: "${author.name|exists}"
          then:
            sections:
              - type: content
                content: |
                  ## 作者

                  **${author.name}**
              - type: conditional
                condition: "${author.email|exists}"
                then:
                  sections:
                    - type: content
                      content: "- 邮箱: ${author.email}"
              - type: conditional
                condition: "${author.url|exists}"
                then:
                  sections:
                    - type: content
                      content: "- 网站: ${author.url}"

output:
  format: markdown
  file_name: "README.md"
  options:
    validate_structure: true
    normalize_whitespace: true
```

这些示例展示了YAML模板系统的强大功能：

1. **结构化定义**: 所有逻辑都在YAML结构中明确定义
2. **类型安全**: 强类型变量定义和验证
3. **模块化组合**: 通过components实现模板的复用
4. **条件逻辑**: 基于变量值动态生成内容
5. **循环处理**: 处理数组和对象的迭代
6. **计算变量**: 基于其他变量计算衍生值
7. **验证机制**: 内置的数据验证和一致性检查

相比传统的Markdown模板，YAML模板提供了更强的逻辑性和更好的可维护性。
