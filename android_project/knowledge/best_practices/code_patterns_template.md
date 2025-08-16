# 代码实现模式库

> **版本**: v1.0  
> **最后更新**: YYYY-MM-DD  
> **维护Agent**: agent-code-reviewer

---

## 代码模式索引

| 模式ID | 模式名称 | 应用场景 | 复杂度 | PyTorch版本 | 性能影响 |
|--------|----------|----------|--------|-------------|----------|
| CP-001 | Lightning模块封装 | 标准训练循环 | 低 | 1.8+ | 无影响 |
| CP-002 | 动态模型构建 | 可配置架构 | 中 | 1.8+ | 轻微开销 |
| CP-003 | 内存高效训练 | 大模型训练 | 高 | 2.0+ | 显著优化 |
| CP-004 | 多GPU分布式 | 大规模训练 | 高 | 1.8+ | 性能提升 |

---

## 核心实现模式

### CP-001: PyTorch Lightning标准封装模式

#### 模式概述
**目标**: 标准化训练循环，减少样板代码
**适用场景**: 90%以上的深度学习项目
**核心价值**: 代码复用性高，易于维护和测试

#### 标准实现
```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict, Tuple

class StandardLightningModule(pl.LightningModule):
    """标准Lightning模块模板
    
    Features:
    - 自动混合精度训练
    - 灵活的优化器配置
    - 内置指标记录
    - 可配置的学习率调度
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        **kwargs
    ):
        super().__init__()
        
        # 保存超参数到checkpoints
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 指标计算
        self.train_acc = pl.metrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = pl.metrics.Accuracy(task="multiclass", num_classes=num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.model(x)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """训练步骤"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算准确率
        acc = self.train_acc(logits, y)
        
        # 记录指标
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return {'loss': loss}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> Dict[str, Any]:
        """验证步骤"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # 计算准确率
        acc = self.val_acc(logits, y)
        
        # 记录指标
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """配置优化器和学习率调度器"""
        # 优化器
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # 学习率调度器
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val/loss'
            }
        }
    
    def configure_callbacks(self) -> list:
        """配置回调函数"""
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor='val/acc',
                mode='max',
                save_top_k=3,
                filename='{epoch}-{val_acc:.3f}'
            ),
            pl.callbacks.EarlyStopping(
                monitor='val/loss',
                patience=10,
                mode='min'
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        ]
        return callbacks
```

#### 使用示例
```python
# 模型定义
model = timm.create_model('resnet50', pretrained=True, num_classes=10)

# Lightning模块
lightning_module = StandardLightningModule(
    model=model,
    num_classes=10,
    learning_rate=1e-3,
    max_epochs=100
)

# 训练器
trainer = pl.Trainer(
    max_epochs=100,
    accelerator='gpu',
    devices=1,
    precision=16,  # 混合精度
    callbacks=lightning_module.configure_callbacks()
)

# 开始训练
trainer.fit(lightning_module, train_dataloader, val_dataloader)
```

---

### CP-002: 动态模型构建模式

#### 模式概述
**目标**: 通过配置文件动态构建不同的模型架构
**适用场景**: 需要实验多种架构的研究项目
**核心价值**: 架构搜索、超参数优化、模型对比

#### 配置驱动的模型构建
```python
from dataclasses import dataclass
from typing import List, Optional, Union
import torch.nn as nn

@dataclass
class LayerConfig:
    """层配置"""
    type: str  # 'conv2d', 'linear', 'attention'
    in_features: Optional[int] = None
    out_features: Optional[int] = None
    kernel_size: Optional[int] = None
    stride: Optional[int] = 1
    padding: Optional[int] = 0
    activation: Optional[str] = 'relu'
    dropout: Optional[float] = 0.0
    
@dataclass 
class ModelConfig:
    """模型配置"""
    name: str
    input_shape: List[int]
    num_classes: int
    layers: List[LayerConfig]

class DynamicModelBuilder:
    """动态模型构建器"""
    
    def __init__(self):
        self.layer_registry = {
            'conv2d': self._build_conv2d,
            'linear': self._build_linear,
            'attention': self._build_attention,
            'residual_block': self._build_residual_block
        }
        
        self.activation_registry = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'swish': nn.SiLU,
            'leaky_relu': nn.LeakyReLU
        }
    
    def build_model(self, config: ModelConfig) -> nn.Module:
        """根据配置构建模型"""
        layers = []
        
        for layer_config in config.layers:
            layer = self._build_layer(layer_config)
            layers.append(layer)
            
            # 添加激活函数
            if layer_config.activation:
                activation = self.activation_registry[layer_config.activation]()
                layers.append(activation)
                
            # 添加Dropout
            if layer_config.dropout > 0:
                layers.append(nn.Dropout(layer_config.dropout))
        
        # 添加分类头
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(layers[-3].out_channels, config.num_classes))
        
        return nn.Sequential(*layers)
    
    def _build_layer(self, config: LayerConfig) -> nn.Module:
        """构建单个层"""
        if config.type not in self.layer_registry:
            raise ValueError(f"Unknown layer type: {config.type}")
        
        return self.layer_registry[config.type](config)
    
    def _build_conv2d(self, config: LayerConfig) -> nn.Module:
        """构建卷积层"""
        return nn.Conv2d(
            in_channels=config.in_features,
            out_channels=config.out_features,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding
        )
    
    def _build_linear(self, config: LayerConfig) -> nn.Module:
        """构建线性层"""
        return nn.Linear(config.in_features, config.out_features)
        
    def _build_attention(self, config: LayerConfig) -> nn.Module:
        """构建注意力层"""
        return nn.MultiheadAttention(
            embed_dim=config.in_features,
            num_heads=8,
            dropout=config.dropout
        )
```

#### 配置文件示例
```yaml
# model_config.yaml
model:
  name: "custom_resnet"
  input_shape: [3, 224, 224]
  num_classes: 10
  layers:
    - type: "conv2d"
      in_features: 3
      out_features: 64
      kernel_size: 7
      stride: 2
      padding: 3
      activation: "relu"
      
    - type: "residual_block"
      in_features: 64
      out_features: 64
      num_blocks: 3
      
    - type: "conv2d"
      in_features: 64
      out_features: 128
      kernel_size: 3
      stride: 2
      padding: 1
      activation: "relu"
      dropout: 0.1
```

---

### CP-003: 内存高效训练模式

#### 模式概述
**目标**: 在有限GPU内存下训练大模型
**适用场景**: 大模型训练、GPU内存不足场景
**核心技术**: 梯度累积、检查点、混合精度

#### 内存优化实现
```python
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional

class MemoryEfficientTrainer:
    """内存高效训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        gradient_accumulation_steps: int = 4,
        use_gradient_checkpointing: bool = True,
        use_mixed_precision: bool = True,
        max_grad_norm: float = 1.0
    ):
        self.model = model
        self.optimizer = optimizer
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.max_grad_norm = max_grad_norm
        
        # 混合精度训练
        if use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # 启用梯度检查点
        if use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """启用梯度检查点以节省内存"""
        def checkpoint_wrapper(module):
            def forward(*args, **kwargs):
                return checkpoint(module.original_forward, *args, **kwargs)
            
            module.original_forward = module.forward
            module.forward = forward
            
        # 对Transformer层启用检查点
        for name, module in self.model.named_modules():
            if 'layer' in name or 'block' in name:
                checkpoint_wrapper(module)
    
    def train_step(self, batch, criterion):
        """内存高效的训练步骤"""
        self.model.train()
        
        # 将batch分成小的sub-batches
        batch_size = len(batch[0])
        sub_batch_size = batch_size // self.gradient_accumulation_steps
        
        total_loss = 0
        self.optimizer.zero_grad()
        
        for i in range(self.gradient_accumulation_steps):
            # 获取sub-batch
            start_idx = i * sub_batch_size
            end_idx = start_idx + sub_batch_size
            sub_batch = [x[start_idx:end_idx] for x in batch]
            
            # 前向传播
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(sub_batch[0])
                    loss = criterion(outputs, sub_batch[1])
                    loss = loss / self.gradient_accumulation_steps
                
                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(sub_batch[0])
                loss = criterion(outputs, sub_batch[1])
                loss = loss / self.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item()
            
            # 清理中间变量以释放内存
            del outputs, loss
            torch.cuda.empty_cache()
        
        # 梯度裁剪
        if self.scaler:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        return total_loss
```

#### 内存监控工具
```python
class MemoryMonitor:
    """GPU内存监控工具"""
    
    @staticmethod
    def get_memory_info():
        """获取GPU内存信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3     # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated': allocated,
                'cached': cached,
                'total': total,
                'free': total - cached
            }
        return None
    
    @staticmethod
    def print_memory_usage(stage: str = ""):
        """打印内存使用情况"""
        info = MemoryMonitor.get_memory_info()
        if info:
            print(f"{stage} - GPU Memory: "
                  f"Allocated: {info['allocated']:.2f}GB, "
                  f"Cached: {info['cached']:.2f}GB, "
                  f"Free: {info['free']:.2f}GB")
```

---

### CP-004: 多GPU分布式训练模式

#### 模式概述
**目标**: 高效利用多GPU资源进行分布式训练
**适用场景**: 大数据集、大模型训练
**核心技术**: DDP、FSDP、模型并行

#### 分布式训练实现
```python
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os

class DistributedTrainer:
    """分布式训练器"""
    
    def __init__(self, backend: str = 'nccl'):
        self.backend = backend
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # 初始化分布式环境
        self._init_distributed()
    
    def _init_distributed(self):
        """初始化分布式环境"""
        if self.world_size > 1:
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size
            )
            torch.cuda.set_device(self.local_rank)
    
    def setup_model(self, model: nn.Module) -> nn.Module:
        """设置分布式模型"""
        model = model.to(self.local_rank)
        
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False  # 优化性能
            )
        
        return model
    
    def setup_dataloader(self, dataset, batch_size: int, shuffle: bool = True):
        """设置分布式数据加载器"""
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle
            )
            shuffle = False  # 使用sampler时不能同时shuffle
        else:
            sampler = None
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader, sampler
    
    def is_main_process(self) -> bool:
        """判断是否为主进程"""
        return self.rank == 0
    
    def save_checkpoint(self, model, optimizer, epoch, path):
        """保存检查点(仅主进程)"""
        if self.is_main_process():
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)
    
    def cleanup(self):
        """清理分布式环境"""
        if self.world_size > 1:
            dist.destroy_process_group()
```

#### 启动脚本
```bash
#!/bin/bash
# distributed_train.sh

export MASTER_ADDR="localhost"
export MASTER_PORT="12355"

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py \
    --batch_size 32 \
    --epochs 100
```

---

## 性能优化模式

### 数据加载优化
```python
class OptimizedDataLoader:
    """优化的数据加载器"""
    
    @staticmethod
    def create_efficient_dataloader(
        dataset,
        batch_size: int,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        persistent_workers: bool = True
    ):
        """创建高效的数据加载器"""
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,  # 加速GPU传输
            prefetch_factor=prefetch_factor,  # 预取数据
            persistent_workers=persistent_workers,  # 保持worker进程
            drop_last=True  # 保持batch大小一致
        )
```

### 模型编译优化(PyTorch 2.0+)
```python
def optimize_model_for_inference(model: nn.Module) -> nn.Module:
    """优化模型用于推理"""
    # 编译模型(PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode='max-autotune')
    
    # 设置为评估模式
    model.eval()
    
    # 预热
    dummy_input = torch.randn(1, 3, 224, 224).cuda()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    return model
```

---

## 代码质量检查清单

### 必须检查项 ✅
- [ ] 所有魔数都定义为常量
- [ ] 函数和类都有类型注解
- [ ] 关键函数有docstring说明
- [ ] 错误处理覆盖主要异常情况
- [ ] 内存泄漏检查(特别是循环引用)
- [ ] GPU内存释放(torch.cuda.empty_cache())

### 性能检查项 ⚡
- [ ] 避免不必要的tensor.cpu()调用
- [ ] 使用in-place操作减少内存分配
- [ ] 批处理而非循环处理
- [ ] 合理使用torch.no_grad()
- [ ] 数据预处理并行化

### 可维护性检查项 🔧
- [ ] 配置和代码分离
- [ ] 模块化设计，职责单一
- [ ] 可复用的组件提取为类
- [ ] 日志记录关键步骤
- [ ] 单元测试覆盖核心逻辑

---

## 反模式警告 ⚠️

### 常见错误模式
```python
# ❌ 错误: 在循环中创建tensor
for i in range(1000):
    x = torch.tensor([i])  # 每次创建新tensor

# ✅ 正确: 预分配tensor
x = torch.zeros(1000)
for i in range(1000):
    x[i] = i

# ❌ 错误: 不必要的GPU-CPU传输
for batch in dataloader:
    loss = criterion(model(batch[0]), batch[1])
    print(f"Loss: {loss.item()}")  # 每次都传输到CPU

# ✅ 正确: 批量记录
losses = []
for batch in dataloader:
    loss = criterion(model(batch[0]), batch[1])
    losses.append(loss.item())
print(f"Average loss: {sum(losses)/len(losses)}")
```

### 内存泄漏模式
```python
# ❌ 错误: 保持对计算图的引用
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []  # 保持对loss tensor的引用
    
    def forward(self, x):
        output = self.linear(x)
        loss = self.criterion(output, target)
        self.losses.append(loss)  # 内存泄漏!
        return output

# ✅ 正确: 只保存标量值
class GoodModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []
    
    def forward(self, x):
        output = self.linear(x)
        loss = self.criterion(output, target)
        self.losses.append(loss.item())  # 只保存数值
        return output
```