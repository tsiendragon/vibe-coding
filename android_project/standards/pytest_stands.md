# PyTest 测试规范

## 核心规则

### 必须 (MUST)
- 为每个新模块创建测试文件
- 每个公开函数至少包含1个正常路径 + 1个错误处理测试
- 使用 `tmp_path` fixture 处理文件I/O测试
- 测试覆盖率 ≥90% (行覆盖 + 分支覆盖)

### 禁止 (MUST NOT)  
- 单元测试中使用 `time.sleep()`
- 未标记 `@pytest.mark.integration` 进行网络调用
- 测试间存在依赖关系
- 使用无意义的测试名称

## 目录结构

```
src/x/y.py → tests/unit/x/test_y.py
src/x/y.py → tests/integration/x/test_y.py  # 需要本地依赖
src/x/y.py → tests/e2e/x/test_y.py          # 需要外部服务
```

## 命名规范

### 文件命名
- `test_<module>.py` 或 `test_<ClassName>.py`

### 函数命名
- `test_<目标>_<行为>[_<条件>]`
- 示例: `test_parse_json_handles_empty_string`

## 测试分层

| 层级 | 用途 | 依赖 | 标记 |
|------|------|------|------|
| unit | 纯逻辑测试 | 无I/O、无网络 | 默认 |
| integration | 本地集成 | 数据库、文件系统 | `@pytest.mark.integration` |
| e2e | 端到端 | 真实外部服务 | `@pytest.mark.e2e` |

### 层级选择决策树
```
纯逻辑测试? → unit/
需要本地文件/DB? → integration/  
需要外部API? → e2e/
```

## 测试模板

```python
"""模块测试文件"""
import pytest
from pathlib import Path
from src.module import TargetClass  # 更新导入路径

class TestTargetClass:
    """TargetClass测试类"""
    
    def test_method_正常路径(self):
        """测试正常操作成功"""
        # Arrange: 准备测试数据
        input_data = {...}
        
        # Act: 执行操作
        result = TargetClass().method(input_data)
        
        # Assert: 验证结果
        assert result.status == "success"
        assert result.value == expected_value
    
    def test_method_处理空输入(self):
        """测试空输入边界情况"""
        with pytest.raises(ValueError, match="输入不能为空"):
            TargetClass().method(None)
```

## 测试内容清单

### 每个函数必测
- ✓ 正常输入输出
- ✓ 边界条件 (空值、极值)
- ✓ 错误路径 (异常处理)
- ✓ 类型验证

### 关键场景必测
- ✓ 并发安全性
- ✓ 幂等性
- ✓ 资源清理
- ✓ 超时处理

## 反模式示例

```python
# ❌ 错误示例
def test_everything():  # 测试过于宽泛
def test_1():          # 无意义命名
assert result          # 断言不具体
time.sleep(1)          # 单元测试中真实延迟

# ✅ 正确示例  
def test_parse_config_invalid_yaml():  # 清晰的命名
    with pytest.raises(ConfigError):   # 具体的断言
        parse_config("invalid: [yaml")
```

## 配置文件

**pytest.ini**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    integration: 需要本地依赖的测试
    e2e: 需要外部服务的测试
```

**conftest.py**
```python
import pytest

@pytest.fixture(autouse=True)
def disable_network_in_unit_tests(monkeypatch, request):
    """单元测试中禁用网络"""
    if "integration" not in request.keywords and "e2e" not in request.keywords:
        def guard(*args, **kwargs):
            raise RuntimeError("单元测试中禁止网络调用")
        monkeypatch.setattr("socket.socket", guard)
```

## CI集成

```yaml
# .github/workflows/test.yml
- name: 运行测试
  run: |
    pytest tests/unit --cov=src --cov-report=term-missing
    pytest tests/integration -m integration
    pytest tests/e2e -m e2e --env=staging
    
- name: 检查覆盖率
  run: |
    coverage report --fail-under=90
```

## 快速参考

| 场景 | 解决方案 |
|------|----------|
| 文件操作 | 使用 `tmp_path` fixture |
| 时间相关 | 使用 `freezegun` 或 mock |
| 外部API | 使用 `responses` 或 `vcr.py` |
| 数据库 | 使用事务回滚或内存数据库 |
| 异步代码 | 使用 `pytest-asyncio` |