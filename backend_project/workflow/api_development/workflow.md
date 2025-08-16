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

## 详细工作流阶段

### Phase 1: API需求分析
**负责**: agent-api-architect  
**起始文档**: `docs/templates/PRD/prd_template.md` → 复制为 `docs/PRD.md`  
**产出文档**: `docs/API_REQUIREMENTS.md`

**详细步骤**:
1. **阅读PRD**: 从`docs/PRD.md`分析业务需求和功能点
2. **需求分解**: 将业务需求转换为API端点需求
3. **端点规划**: 设计REST API端点结构(GET/POST/PUT/DELETE)
4. **数据格式**: 定义请求/响应的JSON Schema
5. **认证策略**: 确定JWT/OAuth2认证方案
6. **更新文档**: 创建`docs/API_REQUIREMENTS.md`记录所有API需求

**Git提交**: 
```bash
git add docs/API_REQUIREMENTS.md
git commit -m "feat(api): define API requirements and endpoint specifications"
```

**成功标准**: API需求清晰、端点设计符合RESTful规范  
**失败处理**: 需求不明确 → 返回与Product Manager确认PRD  
**通知下游**: 完成后通知 `agent-database-engineer` 开始数据库设计

---

### Phase 2: 数据库设计
**负责**: agent-database-engineer  
**协作**: agent-api-architect  
**起始文档**: `docs/API_REQUIREMENTS.md`  
**产出文档**: `docs/DATABASE_DESIGN.md`

**详细步骤**:
1. **需求分析**: 基于`docs/API_REQUIREMENTS.md`分析数据存储需求
2. **ER图设计**: 设计实体关系图，定义表结构
3. **字段定义**: 详细定义每个表的字段、类型、约束
4. **索引策略**: 基于查询模式设计索引
5. **迁移脚本**: 编写Alembic迁移脚本
6. **性能评估**: 预估查询性能，优化设计
7. **更新文档**: 完善`docs/DATABASE_DESIGN.md`

**Git提交**: 
```bash
git add docs/DATABASE_DESIGN.md alembic/versions/
git commit -m "feat(db): design database schema and migration scripts"
```

**成功标准**: 数据模型规范化、查询性能预估<100ms  
**失败处理**: 
- 数据模型不合理 → 重新设计ER图
- 性能预估不达标 → 优化索引策略
**通知下游**: 完成后通知 `agent-api-architect` 进行API规范制定

---

### Phase 3: API规范制定
**负责**: agent-api-architect  
**起始文档**: `docs/API_REQUIREMENTS.md`, `docs/DATABASE_DESIGN.md`  
**产出文档**: `docs/API_SPECIFICATION.md`

**详细步骤**:
1. **OpenAPI编写**: 基于需求和数据库设计编写OpenAPI 3.0规范
2. **端点详化**: 详细定义每个API端点的请求/响应
3. **错误码设计**: 设计统一的错误码体系
4. **认证规范**: 定义JWT token格式和刷新机制
5. **版本控制**: 制定API版本控制策略(v1/v2)
6. **限流规则**: 设计Rate Limiting和请求配额
7. **文档生成**: 生成Swagger UI文档

**Git提交**: 
```bash
git add docs/API_SPECIFICATION.md swagger/
git commit -m "feat(api): complete OpenAPI specification and documentation"
```

**成功标准**: OpenAPI规范完整、可生成Swagger文档  
**失败处理**: 
- 规范不符合OpenAPI标准 → 重新编写规范
- 端点设计冲突 → 与database-engineer协调修改
**通知下游**: 完成后通知 `agent-backend-developer` 开始API开发

---

### Phase 4: 核心API开发
**负责**: agent-backend-developer  
**协作**: agent-database-engineer  
**起始文档**: `docs/API_SPECIFICATION.md`  
**产出代码**: `src/api/` 目录下的实现代码

**详细步骤**:
1. **项目初始化**: 基于FastAPI创建项目结构
2. **数据模型**: 使用SQLAlchemy实现数据库模型
3. **API路由**: 实现所有API端点路由
4. **业务逻辑**: 在Service层实现核心业务逻辑
5. **数据验证**: 使用Pydantic进行请求数据验证
6. **异常处理**: 实现统一的异常处理机制
7. **日志记录**: 添加结构化日志记录
8. **API测试**: 编写基础的API测试

**Git提交** (按模块提交):
```bash
# 数据模型完成
git add src/models/
git commit -m "feat(models): implement SQLAlchemy database models"

# API端点完成
git add src/api/
git commit -m "feat(api): implement core API endpoints with FastAPI"

# 业务逻辑完成
git add src/services/
git commit -m "feat(services): implement business logic layer"
```

**成功标准**: 所有API端点可正常调用、数据验证正确  
**失败处理**: 
- API响应时间>200ms → 优化查询逻辑
- 数据验证失败 → 检查Pydantic模型
- 数据库连接问题 → 检查SQLAlchemy配置
**通知下游**: 完成后通知 `agent-security-auditor` 进行安全审计

---

### Phase 5: 安全加固
**负责**: agent-security-auditor  
**协作**: agent-backend-developer  
**起始代码**: API实现代码  
**产出文档**: `docs/SECURITY_AUDIT.md`

**详细步骤**:
1. **代码审计**: 扫描代码中的安全漏洞
2. **认证加固**: 实现JWT认证和token刷新机制
3. **权限控制**: 实现基于角色的访问控制(RBAC)
4. **输入验证**: 加强所有用户输入的验证和清洗
5. **SQL注入防护**: 检查并修复潜在的SQL注入点
6. **HTTPS配置**: 配置SSL/TLS证书
7. **安全头**: 添加必要的HTTP安全头
8. **渗透测试**: 进行基础的渗透测试

**Git提交**:
```bash
git add src/auth/ src/security/
git commit -m "feat(security): implement JWT authentication and RBAC"

git add src/middleware/
git commit -m "feat(security): add security middleware and input validation"
```

**成功标准**: 通过OWASP Top 10安全检查、无高危漏洞  
**失败处理**: 
- 发现高危漏洞 → 立即修复，重新审计
- 认证机制不完善 → 重新设计JWT流程
**通知下游**: 完成后通知 `agent-api-tester` 开始API测试

---

### Phase 6: API测试
**负责**: agent-api-tester  
**起始代码**: 安全加固后的API代码  
**产出文档**: `docs/TEST_REPORT.md`

**详细步骤**:
1. **测试环境**: 搭建独立的测试环境和测试数据库
2. **单元测试**: 使用pytest编写单元测试，覆盖率≥90%
3. **集成测试**: 测试API端点集成功能
4. **性能测试**: 使用locust进行性能测试
5. **负载测试**: 测试并发处理能力(目标>1000 req/s)
6. **安全测试**: 验证认证、授权、输入验证
7. **边界测试**: 测试极端输入和错误场景
8. **文档验证**: 验证API响应与OpenAPI规范一致

**Git提交**:
```bash
git add tests/
git commit -m "test(api): add comprehensive API test suite with 90%+ coverage"

git add docs/TEST_REPORT.md
git commit -m "docs(test): add API testing report and performance metrics"
```

**成功标准**: 
- 代码覆盖率≥90%
- API响应时间P95<200ms
- 并发处理>1000 req/s
- 所有安全测试通过

**失败处理**: 
- 性能不达标 → 通知backend-developer优化代码
- 测试覆盖率不足 → 补充测试用例
- 安全测试失败 → 通知security-auditor重新加固
**通知下游**: 完成后通知 `agent-devops-engineer` 准备部署

---

### Phase 7: 部署上线
**负责**: agent-devops-engineer  
**协作**: agent-backend-developer  
**起始代码**: 测试通过的API代码  
**产出文档**: `docs/DEPLOYMENT.md`

**详细步骤**:
1. **容器化**: 编写Dockerfile和docker-compose.yml
2. **CI/CD配置**: 配置GitHub Actions或GitLab CI
3. **环境配置**: 设置开发/测试/生产环境变量
4. **数据库部署**: 部署PostgreSQL/MySQL并执行迁移
5. **应用部署**: 部署API服务到Kubernetes/Docker Swarm
6. **负载均衡**: 配置Nginx反向代理和负载均衡
7. **监控配置**: 设置Prometheus + Grafana监控
8. **日志收集**: 配置ELK Stack日志收集
9. **健康检查**: 配置API健康检查端点
10. **回滚方案**: 准备蓝绿部署和快速回滚方案

**Git提交**:
```bash
git add Dockerfile docker-compose.yml .github/workflows/
git commit -m "feat(deploy): add Docker containerization and CI/CD pipeline"

git add k8s/ nginx/
git commit -m "feat(deploy): add Kubernetes manifests and Nginx configuration"

git add monitoring/ logging/
git commit -m "feat(deploy): add monitoring and logging infrastructure"
```

**成功标准**: 
- 服务正常运行，健康检查通过
- 监控指标正常
- 负载测试通过
- 回滚方案可用

**失败处理**: 
- 部署失败 → 检查配置，修复后重新部署
- 性能不达标 → 调整资源配置或优化应用
- 监控异常 → 检查监控配置和指标定义
**通知上游**: 部署成功后通知所有相关Agent项目上线完成

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