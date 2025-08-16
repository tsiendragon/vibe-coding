# Python 代码规范 (Python 3.12)

## 专用标准
- **PyTest**: `backend_project/standards/pytest_stands.md`
- **FastAPI**: `backend_project/standards/fastapi_standards.md`

---

## 核心原则 (MUST/MUST NOT)

| 主题 | MUST | MUST NOT |
|------|------|----------|
| **类型系统** | 公开API全量类型注解；使用PEP 695泛型(`class Foo[T]`)；`@override`标注继承 | 滥用`Any`；无注释的`# type: ignore`；返回裸`dict`/`tuple` |
| **导入副作用** | 库代码导入无副作用；初始化放构造函数 | 在`__init__.py`中做I/O/连接/线程启动 |
| **异常处理** | 自定义异常统一`*Error`；使用`raise ... from e`保持因果链 | `except Exception:`；吞错/降级无告警；异常做控制流 |
| **并发** | I/O用`asyncio.TaskGroup`；CPU用进程池；处理`ExceptionGroup` | 线程共享可变全局状态；未`await`协程 |
| **配置** | 配置集中到单一源(env→配置模型)；依赖注入不读全局 | 散落常量；业务代码直接`os.getenv` |
| **I/O** | 统一文本编码；大文件用流式；文件句柄用上下文管理 | 一次性读巨文件；隐式平台依赖路径 |
| **安全** | 仅受信反序列化(`json`/pydantic)；SQL/命令参数化 | 不可信`pickle`；字符串拼接SQL；`shell=True` |
| **日志** | 在边界层埋点(请求ID/耗时)；结构化键值 | 深层库打印大对象/PII；以日志替代度量 |
| **测试** | 关键路径分支覆盖；固定随机性；外部系统mock | 单测直连网络/DB；顺序耦合用例 |

### 额外禁止
- 写dumpy函数绕过错误
- 超复杂try/except掩盖错误  
- 使用fallback逻辑
- 使用print替代logging

---

## 项目结构

```
src/<package>/
  core/        # 纯领域逻辑(无IO、可纯函数化)
  adapters/    # 外部交互：DB/HTTP/消息/缓存
  services/    # 用例编排；跨adapter/核心拼装
  utils/       # 无状态小工具
```

**规则**: IO、重试、超时、速率限制、幂等、可观测性只出现在`adapters/services`，绝不进入`core`。

---

## 类型系统 (Python 3.12)

```python
# PEP 695: 简洁泛型和类型别名
type Row[K, V] = dict[K, V]

class Box[T]:
    def __init__(self, v: T) -> None: 
        self.v = v
    def get(self) -> T: 
        return self.v

# 协议和覆盖
from typing import Protocol, override

class Reader(Protocol):
    def read(self) -> str: ...

class FileReader:
    @override
    def read(self) -> str: ...
```

**要求**:
- 数据模型: `@dataclass(slots=True, frozen=True)`
- 边界返回显式模型(`TypedDict`/dataclass)，不返回裸dict
- 启用严格mypy检查

---

## 错误处理

```python
class AppError(Exception): ...
class ConfigError(AppError): ...
class ExternalError(AppError): ...
class BusinessError(AppError): ...

# 保持异常链
try:
    external_call()
except RequestError as e:
    raise ExternalError("Service unavailable") from e
```

**规则**:
- 错误语义可区分：配置/外部系统/业务规则
- 在adapter层做有限次重试
- 在服务层聚合为单个业务异常

---

## 并发与超时

```python
import asyncio

async def gather_safe(*aws, timeout: float):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(a) for a in aws]
    # 统一超时控制在调用侧
    return await asyncio.wait_for(tasks, timeout=timeout)
```

**规则**:
- 调用侧统一超时；adapter层重试
- 处理`ExceptionGroup`，不静默丢失失败任务

---

## 安全基线

| 场景 | 规则 |
|------|------|
| 反序列化 | 仅`json`或受控模型；拒绝外部`pickle` |
| 子进程 | `subprocess.run([...], check=True)`参数化 |
| SQL/NoSQL | 参数化查询；统一超时/重试 |
| 日志 | 记录请求ID/耗时；严禁令牌/PII |

---

## 评审门禁

| 类型 | 要求 |
|------|------|
| 代码风格 | Ruff + Black (行宽100) |
| 类型检查 | `mypy --strict`；每个ignore带原因 |
| 测试 | 覆盖率≥90%(含分支)；关键路径必测失败分支 |
| 专用标准 | 测试→pytest_stands.md；PyTorch→pytorch_standards.md |

---

## 工具配置

**pyproject.toml**
```toml
[tool.black]
line-length = 100
target-version = ["py312"]

[tool.ruff]
line-length = 100
target-version = "py312"
extend-select = ["I", "UP", "B", "C90", "T20"]
fix = true

[tool.mypy]
python_version = "3.12"
strict = true
disallow_any_generics = true
warn_return_any = true

[tool.pytest.ini_options]
addopts = "-q --strict-markers --maxfail=1"
testpaths = ["tests"]
```

**coverage要求**
```ini
[run]
branch = True
source = src

[report]
fail_under = 90
show_missing = True
```

---

## 代码模板

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, override
import logging

log = logging.getLogger(__name__)

# 类型定义
type Row[K, V] = dict[K, V]

class Reader(Protocol):
    def read(self) -> str: ...

@dataclass(slots=True, frozen=True)
class Config:
    endpoint: str
    timeout_s: float = 3.0

# Adapter层：外部交互
class FileReader:
    def __init__(self, path: str) -> None:
        self._path = path
    
    def read(self) -> str:
        with open(self._path, "r", encoding="utf-8") as f:
            return f.read()

# Service层：业务编排
class ProcessingService:
    def __init__(self, reader: Reader, cfg: Config) -> None:
        self._reader = reader
        self._cfg = cfg
    
    def process(self) -> str:
        data = self._reader.read()
        log.info("processing_complete", extra={
            "endpoint": self._cfg.endpoint, 
            "size": len(data)
        })
        return data.upper()
```