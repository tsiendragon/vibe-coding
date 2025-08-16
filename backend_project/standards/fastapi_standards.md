# FastAPI Standards

## 目录规范

```bash
your_project/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ alembic.ini               # 数据库迁移配置
├─ configs/
│  └─ default.yaml
├─ scripts/
│  ├─ start.sh
│  ├─ test.sh
│  └─ migrate.sh
├─ src/
│  ├─ api/
│  │  ├─ v1/                 # API版本控制
│  │  │  ├─ endpoints/       # 各模块端点
│  │  │  │  ├─ auth.py
│  │  │  │  ├─ users.py
│  │  │  │  └─ items.py
│  │  │  └─ api.py          # 路由聚合
│  │  └─ deps.py            # 依赖注入
│  ├─ core/
│  │  ├─ config.py          # 配置管理
│  │  ├─ security.py        # 认证授权
│  │  ├─ database.py        # 数据库连接
│  │  └─ exceptions.py      # 异常处理
│  ├─ models/
│  │  ├─ base.py            # SQLAlchemy基类
│  │  ├─ user.py            # 用户模型
│  │  └─ item.py            # 业务模型
│  ├─ schemas/
│  │  ├─ base.py            # Pydantic基类
│  │  ├─ user.py            # 用户模式
│  │  └─ item.py            # 业务模式
│  ├─ services/
│  │  ├─ auth_service.py    # 认证服务
│  │  ├─ user_service.py    # 用户服务
│  │  └─ item_service.py    # 业务服务
│  ├─ utils/
│  │  ├─ auth/              # 认证工具
│  │  ├─ database/          # 数据库工具
│  │  ├─ validation/        # 数据验证
│  │  └─ functions/         # 通用函数
│  ├─ middleware/
│  │  ├─ cors.py            # CORS中间件
│  │  ├─ auth.py            # 认证中间件
│  │  └─ logging.py         # 日志中间件
│  ├─ main.py              # FastAPI应用入口
│  └─ cli.py               # 命令行工具
├─ tests/
│  ├─ pytest.ini
│  ├─ README.md
│  ├─ conftest.py
│  ├─ resources/
│  ├─ unit/
│  │  ├─ api/
│  │  │  └─ v1/
│  │  │     └─ endpoints/
│  │  ├─ core/
│  │  ├─ models/
│  │  ├─ schemas/
│  │  ├─ services/
│  │  └─ utils/
│  ├─ integration/
│  │  ├─ api/
│  │  ├─ database/
│  │  └─ services/
│  └─ e2e/
│     └─ api/
├─ alembic/                 # 数据库迁移文件
│  ├─ versions/
│  └─ env.py
└─ .pre-commit-config.yaml
```

## 应用架构

### 分层架构
- **API层**: 处理HTTP请求响应，参数验证
- **Service层**: 业务逻辑处理
- **Model层**: 数据模型定义
- **Utils层**: 工具函数和共享逻辑

### 依赖注入模式
```python
# src/api/deps.py
from fastapi import Depends
from sqlalchemy.orm import Session
from src.core.database import get_db

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    # 用户验证逻辑
    pass
```

## 配置文件

### 顶层结构（强制）

```yaml
app:
  name: "FastAPI Application"
  version: "1.0.0"
  debug: false
  cors_origins: ["http://localhost:3000"]

database:
  url: "postgresql://user:password@localhost/dbname"
  echo: false
  pool_size: 10
  max_overflow: 20

auth:
  secret_key: "${SECRET_KEY}"
  algorithm: "HS256"
  access_token_expire_minutes: 30

redis:
  url: "redis://localhost:6379"
  decode_responses: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

api:
  prefix: "/api/v1"
  title: "FastAPI Application"
  description: "API Documentation"
```

### MUST / MUST NOT

* **MUST** 使用环境变量管理敏感配置
* **MUST** 支持多环境配置（dev/staging/prod）
* **MUST** 使用Pydantic进行配置验证
* **MUST** 路径使用绝对路径或相对工程根路径
* **MUST NOT** 在配置文件中硬编码密钥
* **MUST NOT** 在代码中硬编码配置值

## 入口脚本（`src/main.py`）

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.api.v1.api import api_router
from src.core.config import settings
from src.middleware.logging import LoggingMiddleware

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        version=settings.VERSION,
    )
    
    # 中间件配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)
    app.add_middleware(LoggingMiddleware)
    
    # 路由注册
    app.include_router(api_router, prefix=settings.API_V1_STR)
    
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
```

## 运行方式

`scripts/start.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# 启动开发服务器
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 生产环境
# gunicorn src.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**常用命令示例**

| 需求         | 命令示例                                    |
| ---------- | --------------------------------------- |
| 启动开发服务器    | `uvicorn src.main:app --reload`        |
| 运行测试       | `pytest`                               |
| 数据库迁移      | `alembic upgrade head`                  |
| 生成迁移文件     | `alembic revision --autogenerate -m "message"` |
| 格式化代码      | `black . && isort .`                   |
| 类型检查       | `mypy .`                               |
| 生成API文档    | 访问 `http://localhost:8000/docs`        |

## 数据库模式

### SQLAlchemy模型
```python
from sqlalchemy import Column, Integer, String, DateTime
from src.models.base import Base

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Pydantic模式
```python
from pydantic import BaseModel, EmailStr
from datetime import datetime

class UserBase(BaseModel):
    email: EmailStr

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        orm_mode = True
```

## 安全标准

* **认证**: 使用JWT令牌
* **授权**: 基于角色的访问控制（RBAC）
* **密码**: 使用bcrypt哈希
* **HTTPS**: 生产环境强制使用
* **CORS**: 严格配置跨域策略
* **输入验证**: 使用Pydantic进行数据验证