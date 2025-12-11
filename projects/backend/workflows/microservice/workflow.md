# Microservice Architecture Workflow

专门针对微服务架构的后端开发工作流，强调服务拆分、通信设计和分布式系统挑战。

## 工作流概述

构建可扩展、高可用的微服务架构，包括服务发现、负载均衡、熔断降级等分布式系统特性。

## 专属Agent团队

| Agent | 角色定位 | 核心输出 | 关键阶段 |
|-------|---------|----------|----------|
| **agent-system-architect** | 系统架构师 | 架构设计、服务边界 | 架构设计、服务拆分 |
| **agent-service-developer** | 服务开发工程师 | 微服务实现 | 服务开发、接口实现 |
| **agent-integration-engineer** | 集成工程师 | 服务通信、消息队列 | 服务集成、通信设计 |
| **agent-reliability-engineer** | 可靠性工程师 | 容错机制、监控方案 | 可靠性设计、故障处理 |
| **agent-data-architect** | 数据架构师 | 数据一致性、分库分表 | 数据架构、事务设计 |
| **agent-platform-engineer** | 平台工程师 | K8s部署、服务网格 | 基础设施、平台搭建 |

## 工作流阶段

### Phase 1: 领域建模与服务边界
**负责**: agent-system-architect  
**产出**: `docs/DOMAIN_MODEL.md`, `docs/SERVICE_BOUNDARIES.md`

- DDD领域驱动设计
- 限界上下文识别
- 服务边界划分
- 依赖关系定义

### Phase 2: 服务接口设计
**负责**: agent-service-developer  
**协作**: agent-system-architect  
**产出**: `docs/SERVICE_CONTRACTS.md`

- gRPC/REST接口定义
- 服务间通信协议
- 版本管理策略
- 向后兼容设计

### Phase 3: 数据架构设计
**负责**: agent-data-architect  
**产出**: `docs/DATA_ARCHITECTURE.md`

- 数据库拆分策略
- 分布式事务方案(Saga/2PC)
- 数据一致性保证
- 缓存策略设计

### Phase 4: 服务开发实现
**负责**: agent-service-developer  
**协作**: agent-data-architect  
**产出**: 各微服务代码

```python
# User Service
class UserService:
    async def create_user(self, user_data: UserCreate) -> User:
        # 创建用户
        user = await self.repository.create(user_data)
        # 发布事件
        await self.event_bus.publish(UserCreatedEvent(user))
        return user

# Order Service  
class OrderService:
    async def create_order(self, order_data: OrderCreate) -> Order:
        # 调用用户服务验证
        user = await self.user_client.get_user(order_data.user_id)
        # 创建订单
        order = await self.repository.create(order_data)
        # 发起支付流程
        await self.payment_client.initiate_payment(order)
        return order
```

### Phase 5: 服务通信与集成
**负责**: agent-integration-engineer  
**产出**: `docs/INTEGRATION_PATTERNS.md`

- 服务发现(Consul/Eureka)
- 负载均衡策略
- 消息队列集成(RabbitMQ/Kafka)
- API网关配置

### Phase 6: 可靠性工程
**负责**: agent-reliability-engineer  
**产出**: `docs/RELIABILITY_DESIGN.md`

- 熔断器模式(Circuit Breaker)
- 重试与超时策略
- 降级方案
- 限流策略

```python
# 熔断器实现
from circuit_breaker import CircuitBreaker

class PaymentServiceClient:
    @CircuitBreaker(failure_threshold=5, recovery_timeout=30)
    async def process_payment(self, payment_data):
        try:
            return await self._http_client.post('/payments', payment_data)
        except Exception as e:
            # 降级处理
            return self._fallback_payment(payment_data)
```

### Phase 7: 平台部署
**负责**: agent-platform-engineer  
**产出**: `docs/DEPLOYMENT_ARCHITECTURE.md`

- Kubernetes部署配置
- Service Mesh(Istio)集成
- 监控体系(Prometheus/Grafana)
- 日志聚合(ELK Stack)

## 微服务特有挑战

### 分布式事务
```python
# Saga模式实现
class OrderSaga:
    async def execute(self, order_data):
        try:
            # Step 1: 创建订单
            order = await self.order_service.create(order_data)
            # Step 2: 扣减库存
            await self.inventory_service.reserve(order.items)
            # Step 3: 处理支付
            payment = await self.payment_service.process(order)
            # Step 4: 确认订单
            await self.order_service.confirm(order.id)
        except Exception as e:
            # 补偿事务
            await self.compensate(order)
```

### 服务间认证
```python
# 服务间JWT认证
class ServiceAuthMiddleware:
    async def __call__(self, request, call_next):
        # 验证服务token
        service_token = request.headers.get('X-Service-Token')
        if not self.verify_service_token(service_token):
            raise HTTPException(401, "Invalid service credentials")
        return await call_next(request)
```

### 分布式追踪
```python
# OpenTelemetry集成
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

class OrderService:
    async def create_order(self, order_data):
        with tracer.start_as_current_span("create_order") as span:
            span.set_attribute("order.user_id", order_data.user_id)
            # 业务逻辑
            result = await self._process_order(order_data)
            span.set_attribute("order.id", result.id)
            return result
```

## 监控指标

### 服务级别指标
- 服务可用性: > 99.9%
- 请求成功率: > 99.5%
- P99延迟: < 500ms
- 错误率: < 0.5%

### 系统级别指标
- CPU使用率: < 70%
- 内存使用率: < 80%
- 网络延迟: < 10ms
- 消息队列延迟: < 100ms

## 部署策略

### 蓝绿部署
```yaml
# Kubernetes蓝绿部署
apiVersion: v1
kind: Service
metadata:
  name: order-service
spec:
  selector:
    app: order-service
    version: green  # 切换到green版本
  ports:
    - port: 80
      targetPort: 8000
```

### 金丝雀发布
```yaml
# Istio金丝雀配置
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: order-service
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: order-service
        subset: v2
  - route:
    - destination:
        host: order-service
        subset: v1
      weight: 90
    - destination:
        host: order-service
        subset: v2
      weight: 10  # 10%流量到新版本
```