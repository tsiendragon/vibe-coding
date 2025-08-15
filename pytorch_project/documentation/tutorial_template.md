# 教程模板

## 入门教程：图像分类项目

### 准备工作
确保已完成环境搭建和数据准备。

### 步骤1: 数据探索
```python
import os
from src.data import DataExplorer

explorer = DataExplorer('data/train')
explorer.show_stats()
explorer.plot_class_distribution()
explorer.show_sample_images(num_samples=10)
```

### 步骤2: 配置训练参数
创建 `my_config.yaml`:
```yaml
model:
  name: "resnet18"
  num_classes: 5

training:
  batch_size: 16
  learning_rate: 0.001
  epochs: 20
```

### 步骤3: 开始训练
```bash
python train.py --config my_config.yaml --name my_first_model
```

### 步骤4: 监控训练过程
打开另一个终端查看日志:
```bash
tail -f logs/my_first_model.log
```

### 步骤5: 评估模型
训练完成后:
```python
from src.models import Model
from src.evaluation import evaluate_model

model = Model.from_pretrained('checkpoints/my_first_model_best.pth')
metrics = evaluate_model(model, 'data/test')
print(f"测试准确率: {metrics['accuracy']:.3f}")
```

### 步骤6: 使用模型预测
```python
# 单张图片预测
result = model.predict('data/test/sample.jpg')
print(f"预测: {result['prediction']}, 置信度: {result['confidence']:.3f}")

# 批量预测
results = model.predict_batch(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

## 进阶教程：自定义数据集训练

### 步骤1: 准备自定义数据集
```python
from src.data import CustomDataset

# 创建自定义数据集
dataset = CustomDataset(
    data_dir='my_custom_data',
    transform=None,  # 使用默认变换
    class_names=['cat', 'dog', 'bird']
)

# 检查数据集
print(f"数据集大小: {len(dataset)}")
print(f"类别数量: {dataset.num_classes}")
```

### 步骤2: 数据增强策略
```python
from torchvision import transforms

# 定义数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = CustomDataset('my_custom_data', transform=train_transform)
```

### 步骤3: 自定义模型架构
```python
import torch.nn as nn
from src.models import BaseModel

class MyCustomModel(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

# 使用自定义模型
model = MyCustomModel(num_classes=3)
```

### 步骤4: 高级训练配置
```yaml
model:
  name: "custom"
  num_classes: 3

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  
  # 学习率调度
  scheduler:
    type: "cosine"
    T_max: 50
    eta_min: 0.00001
  
  # 早停策略
  early_stopping:
    patience: 10
    min_delta: 0.001
    
  # 混合精度训练
  mixed_precision: true
```

## 实用技巧教程

### 技巧1: 可视化训练过程
```python
import matplotlib.pyplot as plt
from src.utils import TrainingVisualizer

visualizer = TrainingVisualizer('logs/training_history.json')
visualizer.plot_loss_curves()
visualizer.plot_accuracy_curves()
visualizer.plot_learning_rate()
```

### 技巧2: 模型诊断
```python
from src.diagnosis import ModelDiagnostics

diagnostics = ModelDiagnostics(model, test_dataloader)

# 检查模型预测
diagnostics.analyze_predictions()

# 可视化错误案例
diagnostics.show_failure_cases(num_cases=10)

# 特征可视化
diagnostics.visualize_features(layer_name='backbone.4')
```

### 技巧3: 超参数调优
```python
from src.tuning import HyperparameterTuner

tuner = HyperparameterTuner()

# 定义搜索空间
search_space = {
    'learning_rate': [0.1, 0.01, 0.001, 0.0001],
    'batch_size': [16, 32, 64],
    'weight_decay': [0, 0.0001, 0.001]
}

# 网格搜索
best_params = tuner.grid_search(search_space, cv_folds=3)
print(f"最佳参数: {best_params}")
```

### 技巧4: 模型压缩和部署
```python
# 模型量化
from src.optimization import ModelQuantizer

quantizer = ModelQuantizer(model)
quantized_model = quantizer.quantize(calibration_data=val_dataloader)

# 转换为ONNX
torch.onnx.export(model, dummy_input, 'model.onnx')

# 性能对比
original_size = quantizer.get_model_size(model)
quantized_size = quantizer.get_model_size(quantized_model)
print(f"压缩率: {original_size / quantized_size:.2f}x")
```

## 故障排除指南

### 常见错误及解决方案

**错误**: `CUDA out of memory`
```python
# 解决方案1: 减少batch size
training:
  batch_size: 8  # 从32减少

# 解决方案2: 梯度累积
training:
  batch_size: 8
  gradient_accumulation_steps: 4  # 等效batch_size=32
```

**错误**: `loss不下降`
```python
# 检查学习率
import torch
from src.utils import LRFinder

lr_finder = LRFinder(model, optimizer, criterion, device)
lr_finder.fit(train_dataloader)
lr_finder.plot()  # 查看最佳学习率
```

**错误**: `验证准确率异常低`
```python
# 检查数据预处理
from src.data import DataValidator

validator = DataValidator()
validator.check_data_consistency(train_dataloader, val_dataloader)
validator.visualize_augmentations(train_dataloader)
```

### 调试工具
```python
# 1. 检查梯度
def check_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"{name}: {grad_norm:.6f}")

# 2. 监控权重变化
def monitor_weights(model, epoch):
    for name, param in model.named_parameters():
        wandb.log({f"weights/{name}": param.data.norm(), "epoch": epoch})

# 3. 数据分布检查
def check_data_distribution(dataloader):
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.tolist())
    
    import collections
    print(collections.Counter(all_labels))
```