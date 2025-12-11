# Standards 标准规范

## 1. 基本结构

### 文件格式
- **文件名**: `.claude/standards/{standard-name}.md` (kebab-case)
- **编码**: UTF-8
- **格式**: Markdown

### 标准元信息
```markdown
# 标准名称
**版本**: v1.0.0
**适用范围**: 描述标准的适用场景和范围
**最后更新**: YYYY-MM-DD
**维护者**: 维护团队或个人
**相关标准**: 引用的其他标准

## 概述
标准的整体描述和目标
```

## 2. 标准分类

### 按领域分类
```markdown
## 流程标准 (Process Standards)
- workflow-standards: 工作流程标准
- quality-gate-standards: 质量门标准
- approval-process-standards: 审批流程标准

## 格式标准 (Format Standards)
- file-format-standards: 文件格式标准
- naming-convention-standards: 命名规范标准
- documentation-standards: 文档标准

## 质量标准 (Quality Standards)
- code-quality-standards: 代码质量标准
- content-quality-standards: 内容质量标准
- performance-standards: 性能标准

## 安全标准 (Security Standards)
- access-control-standards: 访问控制标准
- data-protection-standards: 数据保护标准
- audit-standards: 审计标准

## 技术标准 (Technical Standards)
- api-standards: API设计标准
- integration-standards: 集成标准
- deployment-standards: 部署标准
```

### 按严格程度分类
```markdown
## 强制性标准 (MUST)
- 必须遵守的规则
- 违反将导致流程失败
- 自动化检查和阻止

## 推荐标准 (SHOULD)
- 强烈建议遵守
- 违反时发出警告
- 可以有合理的例外

## 可选标准 (MAY)
- 最佳实践建议
- 提供指导性意见
- 不强制执行
```

## 3. 标准结构

### 基本结构模板
```markdown
# 标准名称

## 1. 目录与产物规范 (Paths & Artifacts)
定义文件路径和产物结构

## 2. 格式规范 (Format Specifications)
定义各种格式的具体要求

## 3. 质量要求 (Quality Requirements)
定义质量标准和评判标准

## 4. 验证规则 (Validation Rules)
定义自动化验证的规则

## 5. 检查工具 (Checking Tools)
提供自动化检查工具

## 6. 示例 (Examples)
提供正确和错误的示例

## 7. 例外处理 (Exception Handling)
定义例外情况的处理方式

## 8. 最佳实践 (Best Practices)
相关的最佳实践建议
```

### 详细内容规范

#### 目录与产物规范
```markdown
## 1. 目录与产物规范 (Paths & Artifacts)

### 目录结构
- 输入目录：`input/`
  - 源文件：`input/source/`
  - 配置文件：`input/config/`
- 输出目录：`output/`
  - 报告：`output/reports/`
  - 产物：`output/artifacts/`
- 临时目录：`temp/`

### 文件命名
- 使用小写字母和连字符
- 包含版本号或时间戳
- 避免特殊字符

### 编码要求
- UTF-8编码
- Unix换行符（LF）
- 无BOM标记

> **统一要求**: 所有文件路径相对于工作空间根目录
```

#### 格式规范
```markdown
## 2. 格式规范 (Format Specifications)

### 文档格式
**MUST**
- 标题使用ATX格式（# ## ###）
- 代码块使用三重反引号
- 链接使用标准Markdown格式

**SHOULD**
- 使用语义化的标题层级
- 提供目录导航
- 包含示例代码

### 数据格式
**JSON格式**
```json
{
  "version": "1.0.0",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "key": "value"
  }
}
```

**YAML格式**
```yaml
version: 1.0.0
timestamp: 2024-01-01T00:00:00Z
data:
  key: value
```
```

#### 质量要求
```markdown
## 3. 质量要求 (Quality Requirements)

### 完整性要求
**MUST**
- 包含所有必需的部分
- 提供完整的配置信息
- 覆盖所有用例场景

**SHOULD**
- 提供使用示例
- 包含故障排除指南
- 维护变更历史

### 准确性要求
**MUST**
- 信息准确无误
- 示例可以正常运行
- 链接有效可访问

### 可读性要求
**SHOULD**
- 使用清晰的语言
- 提供适当的图表
- 结构合理易懂
```
