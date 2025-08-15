# TECH_SPEC.md

> 版本: v1.0
> 最后更新: YYYY-MM-DD
> 负责人: TL (Tech Lead)
> 审核人: PM, Modeling Engineer, QA, Code Reviewer


## 1. 背景与目标
- **背景**: 简述该技术实现是基于什么需求（参考 `PRD.md`），以及为什么需要这个结构。
- **目标**: 明确项目完成后在技术层面要达到什么效果（与 `CONTRACTS.md` 保持一致）。

---

## 2. 模块拆分与职责
| 模块名称 | 职责描述 | 输入 | 输出 | 负责人 |
|----------|----------|------|------|--------|
| `data_loader` | 数据加载与预处理 | CSV路径/字段 | Tensor批次(dict) | XXX |
| `model` | 模型结构定义 | Tensor批次 | logits, loss | XXX |
| `loss` | 定义损失函数 | logits, labels | loss | XXX |
| `metrics` | 定义评估指标 | logits, labels | score(float) | XXX |
| `train_loop` | 训练与验证逻辑 | model, dataloader | checkpoint, logs | XXX |
| `eval` | 测试与推理逻辑 | model, test_data | report, metrics | XXX |
| `utils` | 公共工具函数 | 多种 | 多种 | XXX |

---

## 3. 数据流与调用关系
```mermaid
flowchart LR
    A[CSV Dataset] --> B[data_loader]
    B --> C[model]
    C --> D[loss]
    C --> E[metrics]
    D --> F[train_loop]
    E --> F
    F --> G[checkpoint]
    F --> H[eval]
    H --> I[report]
````

---

## 4. 模块内部结构设计

### 4.1 数据模块 (`data_loader`)

* **类与方法**

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, transforms: Callable):
        ...
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ...
    def __len__(self) -> int:
        ...
```

* **约束**: 输出必须符合 `CONTRACTS.md` 中的 batch 规范

### 4.2 模型模块 (`model`)

* 模型类签名：

```python
class MyModel(torch.nn.Module):
    def __init__(self, num_classes: int):
        ...
    def forward(self, images: Tensor, labels: Optional[Tensor] = None) -> Dict[str, Tensor]:
        ...
```

---

## 5. 关键技术细节与决策

* 使用的框架/库版本（如 PyTorch 2.3, torchvision 0.16）
* 模型初始化策略（如 kaiming\_normal\_）
* 优化器类型与参数（如 AdamW, lr=1e-4）
* 数据增强策略（如 RandAugment, MixUp）
* 训练策略（如 FP16, DDP, gradient accumulation）

---

## 6. 边界条件与异常处理

| 场景       | 期望行为             | 处理方式      |
| -------- | ---------------- | --------- |
| 数据缺失     | 样本丢弃             | WARN 日志记录 |
| 标签越界     | 抛出 ValueError    | CRITICAL  |
| GPU 显存不足 | 自动缩小 batch\_size | WARN      |

---

## 7. 测试与验证计划

* 单元测试模块：`data_loader`, `model`, `loss`, `metrics`
* 集成测试模块：`train_loop`, `eval`
* 回归基线比较：与 `baseline.json` 中的指标比较，容差 ±0.5%

---

## 8. 版本与变更记录

| 版本   | 日期         | 修改内容            | 修改人 |
| ---- | ---------- | --------------- | --- |
| v1.0 | YYYY-MM-DD | 初版              | XXX |
| v1.1 | YYYY-MM-DD | 更新模型结构，调整数据增强策略 | XXX |

## 9. 和CONTAST.md 的对比

| 对比项      | **CONTRACTS.md**                                                           | **TECHSPEC.md**                                                        |
| -------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **核心目的** | 锁定**模块之间的契约**（接口、输入输出、数据格式）                                                | 设计**模块内部结构**和**实现方案**（怎么做）                                               |
| **关注范围** | “**说什么语言**”——模块之间传的是什么、格式如何                                                | “**怎么说话**”——模块怎么拆、怎么调用、怎么实现                                              |
| **内容类型** | 纯接口规范、数据形状、类型、约束、容错规则                                                      | 模块划分、类与函数结构、数据流图、技术选型、边界处理                                               |
| **作用时机** | **Contracts Lock** 阶段定稿，不随实现细节变化                                           | **Minimal Path Spec** 阶段制定，可能随设计迭代                                       |
| **变更频率** | 很低（变了就是需求/接口变动）                                                            | 相对高（实现方式可优化）                                                             |
| **谁来写**  | TL 主导，PM/QA/Modeling 审核                                                    | TL 主导，Modeling 工程师参与，PM/QA/Reviewer 审核                                   |
| **谁来看**  | 所有开发、QA、测试、外部对接方                                                           | 内部开发团队为主，QA/Reviewer 配合                                                  |
| **文件关系** | 上游依赖 `PRD.md`                                                              | 下游依赖 `CONTRACTS.md`，并为 Mechanical Slim-down 提供实现参考                       |
| **例子**   | “模型 forward 输入是 `[B, C, H, W]` 的 float32 张量，输出是 `[B, num_classes]` logits” | “模型文件 `model.py` 包含 `MyModel` 类，使用 ResNet-50 backbone，加入 Attention Head” |


💡 **一句话总结**

* `CONTRACTS.md` → **“我们约定的输入输出格式和规则”**（跨模块保证兼容）
* `TECH_SPEC.md` → **“我们怎么把这些约定实现出来”**（内部设计和实现路径）
