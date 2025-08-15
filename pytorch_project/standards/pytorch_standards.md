# pytorch standards

## 目录规范


```bash
your_project/
├─ pyproject.toml
├─ requirements.txt
├─ README.md
├─ configs/
│  └─ default.yaml
├─ scripts/
│  ├─ train.sh
│  └─ test.sh
├─ src/
│  ├─ data/
│  │  ├─ process/           # 原始→中间件：清洗、split、采样
│  │  ├─ augmentation/      # 增强策略与Compose
│  │  ├─ dataset/           # torch.utils.data.Dataset 实现
│  │  └─ datamodule/        # LightningDataModule（仅装配/Loader）
│  ├─ models/
│  │  ├─ network/           # 模型结构（nn.Module）
│  │  ├─ loss/              # 损失函数封装
│  │  ├─ metrics/           # torchmetrics/自定义指标
│  │  ├─ optimizer/         # 优化器构建（AdamW/Adafactor/…）
│  │  └─ scheduler/         # LR 调度器（Cosine/Step/…）
│  ├─ callbacks/            # PL 回调：ckpt/early_stopping/lr_monitor/自定义
│  ├─ utils/
│  │  ├─ math/              # 数值/采样/统计工具
│  │  ├─ text/              # 文本清洗/分词/正则
│  │  ├─ image/             # 图像IO/变换/可视化
│  │  ├─ tensor/            # 张量操作/对齐/掩码
│  │  └─ functions/         # 纯功能函数（通用，不依赖框架）
│  ├─ tools/                # 额外工具：脚本化工具、数据探查、可视化面板
│  ├─ model.py              # LightningModule（只做前向/优化/日志，依赖上面分层）
│  ├─ train.py              # argparse + OmegaConf 入口（无 LightningCLI）
│  └─ predict.py
├─ tests/
│  ├─ pytest.ini
│  ├─ README.md
│  ├─ conftest.py
│  ├─ plugins/
│  │  └─ net_guard.py
│  ├─ resources/
│  ├─ config/
│  │  └─ integration.yaml
│  ├─ unit/
│  │  ├─ data/
│  │  │  ├─ process/
│  │  │  ├─ augmentation/
│  │  │  ├─ dataset/
│  │  │  └─ datamodule/
│  │  ├─ models/
│  │  │  ├─ network/
│  │  │  ├─ loss/
│  │  │  ├─ metrics/
│  │  │  ├─ optimizer/
│  │  │  └─ scheduler/
│  │  ├─ callbacks/
│  │  ├─ utils/
│  │  │  ├─ math/
│  │  │  ├─ text/
│  │  │  ├─ image/
│  │  │  ├─ tensor/
│  │  │  └─ functions/
│  │  └─ tools/
│  ├─ integration/
│  │  ├─ data/
│  │  ├─ train/
│  │  └─ tools/
│  └─ e2e/
│     └─ train/
└─ .pre-commit-config.yaml

```

## 模块设计


## 配置文件

### 顶层结构（强制）

所有“可实例化对象”统一为 `class_path + init_args`。

```yaml
input_keys: [<str>, ...]             # 可选

data:
  class_path: <dot.path.Class>       # e.g. pkg.data.MyDataModule
  init_args: { ... }

model:
  class_path: <dot.path.Class>       # e.g. pkg.trainer.MyLightningModule
  init_args: { ... }                 # 若未内置优化器/调度器，将自动注入顶层 optimizer/lr_scheduler

optimizer:
  class_path: <dot.path.Optimizer>   # e.g. torch.optim.AdamW / bitsandbytes.optim.AdamW8bit
  init_args: { lr: <float>, ... }

lr_scheduler:
  class_path: <dot.path.Scheduler>   # e.g. torch.optim.lr_scheduler.CosineAnnealingLR
  init_args: { ... }

callback:
  class_path: <dot.path.Scheduler>   # callbacks that used in the pytorch lightining for specifc usage like logging purpose
  init_args: { ... }

trainer:
  logger: { class_path: <...>, init_args: { ... } }  # 可选
  strategy: <str> | <StrategyObj>     # 见 §3
  precision: <"bf16-mixed"|"16-mixed"|"32-true"|...>
  devices: <int|-1|"auto">
  # 其余 Trainer 原生参数按原名透传，如：
  enable_checkpointing: true
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  max_steps: 2000
  log_every_n_steps: 100
  val_check_interval: 250
```
###  MUST / MUST NOT

* **MUST** 所有可实例化对象使用 `class_path + init_args`。
* **MUST** 顶层 `trainer.*` 仅包含 PyTorch Lightning 原生字段或上面定义的 `logger/strategy/precision`。
* **MUST** 路径/文件使用绝对路径或相对工程根路径，不使用 `~`。
* **MUST** 支持 dotlist 覆盖：`trainer.devices=1 optimizer.init_args.lr=5e-5`。
* **MUST NOT** 在 `model.init_args` 与顶层重复定义同名优化器/调度器；若重复，以 **`model.init_args` 优先**。
* **MUST NOT** 在 config 里硬编码账号/密钥。


## 入口脚本（`src/train.py`）

```python
import argparse, importlib, os, torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
try:
    from lightning.pytorch.loggers import WandbLogger
    _WANDB = True
except Exception:
    _WANDB = False
from omegaconf import OmegaConf

def _locate(target: str):
    mod, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(mod), name)

def build_logger(cfg):
    if cfg.exp.logger == "tensorboard":
        return TensorBoardLogger(save_dir=cfg.exp.log_dir, name="default")
    elif cfg.exp.logger == "wandb":
        assert _WANDB, "wandb 未安装"
        return WandbLogger(project=cfg.exp.project, save_dir=cfg.exp.log_dir)
    else:
        raise ValueError(f"unknown logger: {cfg.exp.logger}")

def build_callbacks(cfg):
    ckpt = ModelCheckpoint(
        monitor=cfg.trainer.callbacks.checkpoint.monitor,
        mode=cfg.trainer.callbacks.checkpoint.mode,
        save_top_k=cfg.trainer.callbacks.checkpoint.save_top_k,
        filename=cfg.trainer.callbacks.checkpoint.filename,
    )
    es = EarlyStopping(
        monitor=cfg.trainer.callbacks.early_stopping.monitor,
        mode=cfg.trainer.callbacks.early_stopping.mode,
        patience=cfg.trainer.callbacks.early_stopping.patience,
    )
    lr = LearningRateMonitor(logging_interval=cfg.trainer.callbacks.lr_monitor.logging_interval)
    return [ckpt, es, lr]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--fast_dev_run", type=int, default=0)
    # 允许 dotlist 覆盖：trainer.max_epochs=50 data.params.batch_size=64
    args, unknown = parser.parse_known_args()

    base = OmegaConf.load(args.config)
    overrides = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(base, overrides)

    # 复现
    seed_everything(cfg.exp.seed, workers=True)

    # 模型与数据
    ModelCls = _locate(cfg.model.target)
    DataCls  = _locate(cfg.data.target)
    model = ModelCls(**cfg.model.params)
    datamodule = DataCls(**cfg.data.params)

    # Logger & Callbacks
    logger = build_logger(cfg)
    callbacks = build_callbacks(cfg)

    # 可选：bf16 -> 16-mixed 自动降级（无 GPU/bf16 时）
    precision = cfg.trainer.precision
    if precision == "bf16-mixed":
        if not torch.cuda.is_available():
            precision = "16-mixed"

    trainer = Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        precision=precision,
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        deterministic=cfg.trainer.deterministic,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=bool(args.fast_dev_run),
    )

    ckpt_path = cfg.exp.ckpt_path if (cfg.exp.resume and cfg.exp.ckpt_path) else None
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    trainer.test(model=model, datamodule=datamodule)

if __name__ == "__main__":
    # 轻量性能优先
    torch.set_float32_matmul_precision("high")
    main()
```

## 运行方式与覆盖（兼容 dotlist）

`scripts/train.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
python -m src.train \
  --config configs/default.yaml \
  trainer.devices=auto trainer.strategy=ddp \
  trainer.max_epochs=100 data.params.batch_size=128
```

**常用覆盖示例**

| 需求         | 覆盖示例                                                                         |
| ---------- | ---------------------------------------------------------------------------- |
| 快速冒烟       | `--fast_dev_run=1`                                                           |
| 调整 epoch   | `trainer.max_epochs=50`                                                      |
| 单机单卡       | `trainer.devices=1 trainer.strategy=auto`                                    |
| AMP16      | `trainer.precision=16-mixed`                                                 |
| 恢复训练       | `exp.resume=true exp.ckpt_path=logs/default/version_0/checkpoints/last.ckpt` |
| 改 batch 大小 | `data.params.batch_size=64`                                                  |
