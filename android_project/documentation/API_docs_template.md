# API 文档模板

## 模型 API

### Model
主模型类，提供训练和推理功能。

```python
class Model(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True)
```

**参数:**
- `num_classes` (int): 分类数量
- `pretrained` (bool): 是否使用预训练权重

**示例:**
```python
model = Model(num_classes=10, pretrained=True)
```

### Model.forward()
前向传播方法。

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

**参数:**
- `x` (torch.Tensor): 输入张量，形状为 (B, C, H, W)

**返回:**
- `torch.Tensor`: 输出logits，形状为 (B, num_classes)

**示例:**
```python
input_tensor = torch.randn(32, 3, 224, 224)
output = model(input_tensor)  # shape: (32, 10)
```

### Model.predict()
高层预测接口。

```python
def predict(self, x: Union[torch.Tensor, np.ndarray, str]) -> Dict[str, Any]
```

**参数:**
- `x`: 输入数据，支持张量、数组或图片路径

**返回:**
- `Dict[str, Any]`: 预测结果
  - `prediction` (int): 预测类别
  - `confidence` (float): 置信度
  - `probabilities` (List[float]): 各类别概率

**示例:**
```python
result = model.predict('path/to/image.jpg')
print(f"预测类别: {result['prediction']}")
print(f"置信度: {result['confidence']:.3f}")
```

## 数据处理 API

### DataLoader
数据加载器类。

```python
class DataLoader:
    def __init__(self, 
                 dataset_path: str,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4)
```

**参数:**
- `dataset_path` (str): 数据集路径
- `batch_size` (int): 批大小，默认32
- `shuffle` (bool): 是否打乱数据，默认True
- `num_workers` (int): 工作线程数，默认4

### DataLoader.get_dataloader()
获取PyTorch DataLoader对象。

```python
def get_dataloader(self) -> torch.utils.data.DataLoader
```

**返回:**
- `torch.utils.data.DataLoader`: 数据加载器

**示例:**
```python
loader = DataLoader('data/train', batch_size=64)
dataloader = loader.get_dataloader()

for batch_idx, (data, target) in enumerate(dataloader):
    # 训练逻辑
    pass
```

## 训练 API

### Trainer
训练器类。

```python
class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 criterion: nn.Module,
                 device: str = 'cuda')
```

### Trainer.fit()
开始训练。

```python
def fit(self, epochs: int, save_path: str = None) -> Dict[str, List[float]]
```

**参数:**
- `epochs` (int): 训练轮数
- `save_path` (str, optional): 模型保存路径

**返回:**
- `Dict[str, List[float]]`: 训练历史
  - `train_loss`: 训练损失历史
  - `val_loss`: 验证损失历史
  - `val_acc`: 验证准确率历史

**示例:**
```python
trainer = Trainer(model, train_loader, val_loader, optimizer, criterion)
history = trainer.fit(epochs=50, save_path='checkpoints/model.pth')

print(f"最佳验证准确率: {max(history['val_acc']):.3f}")
```

## 工具函数 API

### load_checkpoint()
加载模型检查点。

```python
def load_checkpoint(model: nn.Module, 
                   checkpoint_path: str, 
                   device: str = 'cuda') -> nn.Module
```

**参数:**
- `model` (nn.Module): 模型实例
- `checkpoint_path` (str): 检查点文件路径  
- `device` (str): 设备类型，默认'cuda'

**返回:**
- `nn.Module`: 加载权重后的模型

### save_checkpoint()
保存模型检查点。

```python
def save_checkpoint(model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   path: str) -> None
```

**参数:**
- `model` (nn.Module): 模型
- `optimizer` (torch.optim.Optimizer): 优化器
- `epoch` (int): 当前轮数
- `loss` (float): 当前损失
- `path` (str): 保存路径

**示例:**
```python
# 保存
save_checkpoint(model, optimizer, epoch=10, loss=0.5, path='checkpoint.pth')

# 加载
model = Model(num_classes=10)
model = load_checkpoint(model, 'checkpoint.pth')
```

## 错误处理

### ModelError
模型相关错误。

```python
class ModelError(Exception):
    """模型加载或推理时发生的错误"""
    pass
```

### DataError  
数据处理错误。

```python
class DataError(Exception):
    """数据加载或预处理时发生的错误"""
    pass
```

**常见错误处理:**
```python
try:
    result = model.predict(input_data)
except ModelError as e:
    print(f"模型错误: {e}")
except DataError as e:
    print(f"数据错误: {e}")
```