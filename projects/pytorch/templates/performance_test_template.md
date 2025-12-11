# 性能测试模板

## 性能指标定义

### 推理性能
- **延迟 (Latency)**: 单次推理时间 < 100ms
- **吞吐量 (Throughput)**: 每秒处理样本数 >= 100
- **批处理效率**: batch_size=32时延迟 < 1s
- **内存使用**: 推理峰值内存 < 2GB

### 训练性能  
- **训练速度**: 每个epoch时间 < 300s
- **GPU利用率**: 训练时GPU使用率 > 80%
- **内存效率**: 训练峰值内存 < 8GB
- **收敛速度**: 10个epoch内达到85%准确率

## 性能测试用例

### 延迟测试
```python
@pytest.mark.performance
def test_inference_latency():
    model.eval()
    input_data = torch.randn(1, 3, 224, 224)
    
    # 预热
    for _ in range(10):
        _ = model(input_data)
    
    # 测试
    start_time = time.time()
    with torch.no_grad():
        output = model(input_data)
    latency = time.time() - start_time
    
    assert latency < 0.1, f"延迟 {latency:.3f}s 超过100ms"
```

### 吞吐量测试
```python
def test_throughput():
    batch_sizes = [1, 8, 16, 32]
    for batch_size in batch_sizes:
        input_data = torch.randn(batch_size, 3, 224, 224)
        
        start_time = time.time()
        output = model(input_data)
        duration = time.time() - start_time
        
        throughput = batch_size / duration
        assert throughput >= 100, f"批大小{batch_size}吞吐量{throughput:.1f} < 100"
```

### 内存测试
```python
def test_memory_usage():
    torch.cuda.reset_peak_memory_stats()
    input_data = torch.randn(32, 3, 224, 224).cuda()
    
    output = model(input_data)
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    
    assert peak_memory < 2.0, f"内存使用{peak_memory:.2f}GB超过2GB"
```

## 基准测试

### 性能基准
```python
class PerformanceBenchmark:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
    def benchmark_inference(self, input_shape, num_runs=100):
        input_data = torch.randn(*input_shape).to(self.device)
        
        # 预热
        for _ in range(10):
            _ = self.model(input_data)
        
        # 基准测试
        torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            output = self.model(input_data)
            
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        return {
            'avg_latency': total_time / num_runs,
            'throughput': input_shape[0] * num_runs / total_time,
            'memory_peak': torch.cuda.max_memory_allocated() / 1024**3
        }
```

### 比较基准
```python
def test_performance_regression():
    current_metrics = benchmark_current_model()
    baseline_metrics = load_baseline_metrics()
    
    # 延迟不能增加超过10%
    assert current_metrics['latency'] <= baseline_metrics['latency'] * 1.1
    
    # 吞吐量不能降低超过10%  
    assert current_metrics['throughput'] >= baseline_metrics['throughput'] * 0.9
```

## 压力测试

### 长时间运行
```python
def test_long_running_stability():
    start_time = time.time()
    num_iterations = 1000
    
    for i in range(num_iterations):
        input_data = torch.randn(1, 3, 224, 224)
        output = model(input_data)
        
        # 每100次检查内存
        if i % 100 == 0:
            memory_usage = torch.cuda.memory_allocated() / 1024**3
            assert memory_usage < 2.0, f"第{i}次迭代内存泄漏: {memory_usage:.2f}GB"
```

### 并发测试
```python
def test_concurrent_inference():
    import concurrent.futures
    
    def single_inference():
        input_data = torch.randn(1, 3, 224, 224)
        return model(input_data)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(single_inference) for _ in range(20)]
        results = [f.result() for f in futures]
    
    assert len(results) == 20, "并发推理失败"
```

## 性能分析

### 性能分析工具
```python
def profile_model():
    from torch.profiler import profile, ProfilerActivity
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        output = model(input_data)
    
    # 分析结果
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    
    # 保存详细报告
    prof.export_chrome_trace("trace.json")
```

## 性能报告模板

```markdown
# 性能测试报告

## 测试环境
- GPU: NVIDIA RTX 3090
- CUDA: 11.8
- PyTorch: 2.0.1

## 测试结果
| 指标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 单次推理延迟 | <100ms | 85ms | ✅ |
| 批处理吞吐量 | >100/s | 120/s | ✅ |
| 训练内存峰值 | <8GB | 6.2GB | ✅ |
| GPU利用率 | >80% | 92% | ✅ |

## 性能瓶颈分析
1. **数据加载**: 占用15%训练时间，建议增加num_workers
2. **模型计算**: GPU利用率良好，无明显瓶颈
3. **内存管理**: 内存使用平稳，无泄漏

## 优化建议
1. 使用混合精度训练减少内存使用
2. 优化数据预处理pipeline
3. 考虑模型量化提升推理速度
```