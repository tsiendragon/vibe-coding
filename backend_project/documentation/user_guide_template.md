# 用户使用指南

## 环境搭建

### 系统要求
- Python 3.8+
- 8GB+ 内存
- NVIDIA GPU (可选，用于加速训练)

### 安装步骤
```bash
# 1. 克隆仓库
git clone <repository_url>
cd <project_name>

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证安装
python -c "import torch; print(torch.__version__)"
```

## 数据准备

### 数据格式
支持的数据格式:
- **图像**: JPG, PNG, BMP
- **标注**: JSON, CSV, TXT

### 数据组织
```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
├── val/
└── test/
```

### 数据预处理
```python
from src.data import DataProcessor

processor = DataProcessor()
processor.preprocess_dataset('data/raw', 'data/processed')
```

## 模型训练

### 基础训练
```bash
python train.py --config configs/basic_config.yaml
```

### 自定义配置
编辑 `configs/train_config.yaml`:
```yaml
model:
  name: "resnet50"
  num_classes: 10

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 50

data:
  train_path: "data/train"
  val_path: "data/val"
```

### 恢复训练
```bash
python train.py --resume checkpoints/model_epoch_10.pth
```

## 模型推理

### 单张图片推理
```python
from src.models import Model

# 加载模型
model = Model.from_pretrained('checkpoints/best_model.pth')

# 推理
result = model.predict('path/to/image.jpg')
print(f"预测类别: {result['prediction']}")
print(f"置信度: {result['confidence']:.3f}")
```

### 批量推理
```bash
python inference.py \
    --model checkpoints/best_model.pth \
    --input data/test \
    --output results.json
```

### API推理
```python
import requests

response = requests.post('http://localhost:8000/predict', 
                        files={'image': open('test.jpg', 'rb')})
result = response.json()
```

## 模型评估

### 性能评估
```bash
python evaluate.py \
    --model checkpoints/best_model.pth \
    --dataset data/test \
    --metrics accuracy precision recall f1
```

### 性能分析
```python
from src.evaluation import ModelEvaluator

evaluator = ModelEvaluator(model)
metrics = evaluator.evaluate(test_dataloader)

print(f"准确率: {metrics['accuracy']:.3f}")
print(f"精确率: {metrics['precision']:.3f}")
print(f"召回率: {metrics['recall']:.3f}")
```

## 配置参数

### 模型配置
```yaml
model:
  name: "resnet50"          # 模型架构
  num_classes: 10           # 分类数量
  pretrained: true          # 是否使用预训练权重
  dropout: 0.5              # Dropout率
```

### 训练配置
```yaml
training:
  batch_size: 32            # 批大小
  learning_rate: 0.001      # 学习率
  weight_decay: 0.0001      # 权重衰减
  epochs: 100               # 训练轮数
  early_stopping: 10        # 早停轮数
```

### 数据配置
```yaml
data:
  image_size: [224, 224]    # 图片尺寸
  normalize: true           # 是否标准化
  augmentation: true        # 是否数据增强
  num_workers: 4            # 数据加载线程数
```

## 常见问题

### 训练相关
**Q: 训练过程中显存不足？**
A: 减少batch_size或使用梯度累积:
```yaml
training:
  batch_size: 16  # 从32减少到16
  gradient_accumulation_steps: 2
```

**Q: 模型不收敛？**
A: 调整学习率和优化器:
```yaml
training:
  learning_rate: 0.0001  # 降低学习率
  optimizer: "adamw"     # 尝试不同优化器
```

### 推理相关
**Q: 推理速度慢？**
A: 使用批量推理和模型优化:
```python
# 批量处理
batch_results = model.predict_batch(image_list)

# 转换为torchscript
model = torch.jit.script(model)
```

**Q: 预测结果不准确？**
A: 检查数据预处理和模型版本:
```python
# 确保预处理一致
processor = DataProcessor()
image = processor.preprocess(raw_image)
```

### 部署相关
**Q: 如何部署模型服务？**
A: 使用提供的Docker配置:
```bash
docker build -t model-service .
docker run -p 8000:8000 model-service
```

## 性能优化

### 训练加速
- 使用混合精度训练
- 增加数据加载worker数量
- 使用分布式训练

### 推理加速
- 模型量化
- 批量推理
- 使用TensorRT (NVIDIA GPU)

### 内存优化
- 梯度累积减少batch size
- 梯度检查点技术
- 模型并行化

## 监控和日志

### 训练监控
使用Weights & Biases:
```bash
pip install wandb
wandb login
python train.py --use_wandb
```

### 日志分析
```python
# 分析训练日志
from src.utils import LogAnalyzer

analyzer = LogAnalyzer('logs/train.log')
analyzer.plot_metrics()
```