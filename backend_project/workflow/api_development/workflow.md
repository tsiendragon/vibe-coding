# API Development Workflow

专门针对FastAPI后端API开发的工作流，从需求分析到API部署的完整流程。

## 工作流概述

这个工作流专注于构建RESTful API服务，强调API设计、数据库架构、性能优化和安全性。

## 专属Agent团队

| Agent | 角色定位 | 核心输出 | 关键阶段 |
|-------|---------|----------|----------|
| **agent-api-architect** | API架构设计师 | API规范、系统架构 | API设计、架构评审 |
| **agent-backend-developer** | 后端开发工程师 | API实现、业务逻辑 | 开发实现、优化 |
| **agent-database-engineer** | 数据库工程师 | 数据模型、查询优化 | 数据库设计、优化 |
| **agent-security-auditor** | 安全审计员 | 安全报告、漏洞修复 | 安全审计、加固 |
| **agent-api-tester** | API测试工程师 | 测试用例、性能报告 | 测试验证、压测 |
| **agent-devops-engineer** | DevOps工程师 | 部署脚本、CI/CD | 部署、监控 |

## 工作流阶段

### Phase 1: API需求分析
**负责**: agent-api-architect  
**产出**: `docs/API_REQUIREMENTS.md`

- 分析业务需求
- 定义API端点
- 设计请求/响应格式
- 确定认证授权策略

### Phase 2: 数据库设计
**负责**: agent-database-engineer  
**协作**: agent-api-architect  
**产出**: `docs/DATABASE_DESIGN.md`

- 设计数据模型
- 定义表关系
- 创建索引策略
- 编写迁移脚本

### Phase 3: API规范制定
**负责**: agent-api-architect  
**产出**: `docs/API_SPECIFICATION.md`

- OpenAPI规范定义
- 错误码设计
- 版本控制策略
- Rate limiting规则

### Phase 4: 核心API开发
**负责**: agent-backend-developer  
**协作**: agent-database-engineer  
**产出**: API端点代码

- 实现CRUD操作
- 业务逻辑开发
- 数据验证层
- 异常处理

### Phase 5: 安全加固
**负责**: agent-security-auditor  
**协作**: agent-backend-developer  
**产出**: `docs/SECURITY_AUDIT.md`

- JWT认证实现
- 权限控制(RBAC)
- SQL注入防护
- XSS/CSRF防护

### Phase 6: API测试
**负责**: agent-api-tester  
**产出**: `docs/TEST_REPORT.md`

- 单元测试
- 集成测试
- 性能测试
- 负载测试

### Phase 7: 部署上线
**负责**: agent-devops-engineer  
**协作**: agent-backend-developer  
**产出**: `docs/DEPLOYMENT.md`

- Docker容器化
- CI/CD配置
- 监控告警设置
- 日志收集配置

## 关键决策点

| 决策点 | 决策者 | 通过标准 |
|--------|--------|----------|
| API设计评审 | agent-api-architect | RESTful规范、性能指标 |
| 数据库设计评审 | agent-database-engineer | 规范化、查询性能 |
| 安全审计 | agent-security-auditor | OWASP标准 |
| 性能验收 | agent-api-tester | 响应时间<200ms、并发>1000 |
| 部署批准 | agent-devops-engineer | 健康检查通过、回滚方案就绪 |

## 质量标准

### API设计标准
- RESTful规范遵循度: 100%
- API文档完整性: 100%
- 错误处理覆盖: 100%

### 性能标准
- API响应时间: P95 < 200ms
- 数据库查询: < 100ms
- 并发处理: > 1000 req/s

### 安全标准
- 认证机制: JWT with refresh token
- 加密传输: HTTPS only
- 输入验证: 100%覆盖

### 测试标准
- 代码覆盖率: ≥ 90%
- API端点测试: 100%
- 性能测试: 全部关键接口