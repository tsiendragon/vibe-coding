# 测试策略模板

## 测试层级

### Unit Tests (90%+ 覆盖率)
**目标**: 验证单个函数/类逻辑正确性
```python
def test_model_forward_pass():
    model = MyModel()
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    assert output.shape == (1, 10)
    assert not torch.isnan(output).any()
```

### Integration Tests
**目标**: 验证模块间交互和数据流
```python 
def test_training_pipeline():
    dataloader = create_test_dataloader()
    model = create_test_model()
    trainer = create_test_trainer(model)
    trainer.fit(dataloader)
    assert trainer.logged_metrics['val_acc'] > 0.8
```

### End-to-End Tests  
**目标**: 验证完整用户场景
```python
def test_inference_workflow():
    # 加载模型 -> 预处理数据 -> 推理 -> 后处理
    result = run_inference('test_image.jpg')
    assert result['confidence'] > 0.9
    assert 'prediction' in result
```

## ML特定测试

### 模型测试
- **形状测试**: 输入输出tensor形状正确
- **数值稳定性**: 无NaN/Inf值产生
- **梯度测试**: 反向传播计算正确
- **确定性测试**: 固定seed结果可重现

### 数据测试
- **数据质量**: 无损坏/缺失数据
- **增强一致性**: 数据增强不改变标签
- **批次测试**: 不同batch size结果一致
- **内存泄漏**: 长时间运行内存稳定

### 性能测试
```python
def test_inference_latency():
    start_time = time.time()
    result = model(test_input)
    latency = time.time() - start_time
    assert latency < 0.1  # 100ms
```

## 测试数据策略

### 测试数据类型
- **最小样本**: 验证基本功能
- **典型样本**: 常见使用场景  
- **边界样本**: 极值和边界条件
- **异常样本**: 错误输入和异常情况

### 数据生成
```python
# 合成测试数据
def create_test_dataset(size=100):
    return torch.randn(size, 3, 224, 224)

# 固定测试集
TEST_SAMPLES = [
    'data/test/normal_case.jpg',
    'data/test/edge_case.jpg', 
    'data/test/error_case.jpg'
]
```

## 自动化测试

### CI/CD集成
```yaml
# pytest配置
pytest:
  - tests/unit/ --cov=src --cov-report=xml
  - tests/integration/ --timeout=300
  - tests/e2e/ --slow
```

### 回归测试
- 每次代码变更运行完整测试套件
- 性能回归检测
- 模型准确率回归检测

## 测试环境

### 环境要求
- **CPU测试**: 基础功能验证
- **GPU测试**: 性能和训练测试
- **多GPU测试**: 分布式训练验证

### Mock和Stub
```python
@pytest.fixture
def mock_expensive_computation():
    with patch('src.module.expensive_function') as mock:
        mock.return_value = expected_result
        yield mock
```