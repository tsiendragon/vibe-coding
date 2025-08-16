# Backend API 重写型工作流设计

这个工作流基于现有的系统需求或参考项目，提取核心业务逻辑，然后设计清晰的API架构和后端服务，重新实现更加模块化、易维护的FastAPI后端项目。需要设计不同的agent赋予他们的角色定位和特长，以及项目不同阶段不同agent之间的协作方式。

## Agent 列表

| Agent | 核心职责 | 主要产出 | 关键技能 | 关键工作阶段 |
|-------|----------|----------|----------|-------------|
| **agent-product-manager** | 需求分析、API产品设计、功能验收 | `docs/PRD/PRD.md`、API验收报告 | 业务理解、API设计、需求管理 | 需求映射、功能验收 |
| **agent-tech-lead** | 技术架构设计、技术决策、项目协调、最终交付决策 | `docs/TECH_SPEC/TECH_SPEC.md`、`docs/TODO/TODO.md`、架构评估、交付决策 | 后端架构、微服务设计、技术选型、团队领导 | 架构设计、TECH_SPEC评审、API评估、服务质量评估、项目交付决策 |
| **agent-researcher** | 技术调研、架构可行性分析、最佳实践研究 | `docs/research/architecture_review.md`、`docs/research/recommendations.md`、技术审核报告 | API设计研究、后端技术趋势、架构模式 | 技术调研、TECH_SPEC评审、架构评估、技术一致性审核 |
| **agent-backend-engineer** | API实现、业务逻辑开发、数据库设计 | 核心API代码、服务模块代码、模块README | FastAPI开发、数据库设计、后端优化 | API原型实现、业务逻辑实现、所有服务模块开发 |
| **agent-code-reviewer** | 代码质量审核、标准检查、持续监控 | 代码审核报告、质量改进建议 | 代码审查、编程规范、FastAPI最佳实践 | 代码开发全程、测试代码审查、最终代码审核 |
| **agent-qa-engineer** | 测试用例编写、API测试、性能测试 | 测试代码、API测试报告、性能评估报告 | API测试、pytest规范、性能分析、集成测试 | 服务测试、集成测试、性能测试、API验收测试、质量验收 |
| **agent-docs-writer** | API文档、技术文档、文档体系构建 | API文档、项目README.md、部署指南 | 技术写作、API文档、知识整理 | 最终文档生成 |

## Backend项目特有的工作阶段

### API原型开发

**前置条件**: TECH_SPEC审核通过，API设计方案确定

1. **agent-tech-lead** 与 agent-backend-engineer 协作制定API原型开发计划：
   - 核心API端点设计：用户认证、核心业务接口、数据查询接口
   - 数据模型设计：Pydantic模型、SQLAlchemy模型
   - 验证目标：API响应正确、基本认证可用、数据库连接正常

2. **agent-backend-engineer** 实现核心API原型：
   - 实现最小可行的API端点
   - 建立基本的数据模型
   - 配置基本的认证机制
   - 实现基础的错误处理

3. **agent-qa-engineer** 进行API原型测试：
   - API端点可访问性测试
   - 数据格式验证测试
   - 基本性能测试

### 数据库设计阶段

**前置条件**: API原型验证通过

1. **agent-backend-engineer** 进行完整数据库设计：
   - 设计数据表结构和关系
   - 创建数据库迁移脚本
   - 建立数据访问层(Repository模式)
   - 实现数据验证和约束

2. **agent-code-reviewer** 审核数据库设计：
   - 数据模型合理性
   - 查询性能优化
   - 数据一致性保证

### 服务层开发阶段

**前置条件**: 数据库设计完成

1. **agent-backend-engineer** 实现业务服务层：
   - 用户服务：注册、登录、权限管理
   - 业务服务：核心业务逻辑实现
   - 数据服务：数据查询、聚合、缓存
   - 通知服务：消息推送、邮件通知

2. **agent-qa-engineer** 进行服务层测试：
   - 单元测试：每个服务方法测试
   - 集成测试：服务间协作测试
   - 业务逻辑测试：复杂业务场景验证

## FastAPI特有的质量标准

### API设计标准
- RESTful设计原则
- 统一的响应格式
- 完整的错误处理
- API版本控制策略
- 请求/响应数据验证

### 性能标准
- API响应时间 < 200ms (P95)
- 数据库查询优化
- 合理的缓存策略
- 并发请求处理能力

### 安全标准
- JWT认证机制
- HTTPS强制使用
- 输入数据验证和清理
- SQL注入防护
- CORS配置正确

### 测试标准
- API端点100%覆盖
- 业务逻辑测试覆盖率≥90%
- 集成测试覆盖关键流程
- 性能测试验证响应时间
- 安全测试验证漏洞防护

## 工具和技术栈

### 开发工具
- FastAPI框架进行API开发
- SQLAlchemy进行数据库ORM
- Alembic进行数据库迁移管理
- Pydantic进行数据验证
- Pytest进行测试

### 部署和运维
- Docker容器化部署
- PostgreSQL/MySQL数据库
- Redis缓存服务
- Nginx反向代理
- 日志监控和错误跟踪

### 开发规范
- 遵循 `docs/standards/fastapi_standards.md` - FastAPI开发规范
- 遵循 `docs/standards/pycode_standards.md` - Python编码标准  
- 遵循 `docs/standards/pytest_stands.md` - pytest测试标准

### Git提交规范
- API实现完成: `feat(api): implement [endpoint_name] endpoint`
- 数据库设计完成: `feat(db): add [model_name] model and migration`
- 服务实现完成: `feat(service): implement [service_name] business logic`
- 测试完成: `test(api): add comprehensive tests for [feature_name]`
- 性能优化: `perf(api): optimize [specific_optimization]`