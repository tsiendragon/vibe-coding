# Folder Design

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
│  ├─ model.py        # LightningModule（只做前向/优化/日志，依赖上面分层）
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