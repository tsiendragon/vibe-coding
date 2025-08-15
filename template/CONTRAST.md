# CONTRACTS.md

> 版本: v1.0
> 最后更新: YYYY-MM-DD
> 负责人: TL (Tech Lead)
> 审核人: PM, Modeling Engineer, QA, Code Reviewer

---

## 1. 数据源与映射 (Data Source & Mapping)
| CSV列名 / 字段 | 数据类型 | 预处理 | 张量类型 | 张量形状 | 备注 |
|----------------|----------|--------|----------|----------|------|
| `image_path`   | str      | load+resize(224,224), normalize(0.5,0.5) | float32 | [C,H,W] | RGB, C=3 |
| `label`        | int      | one-hot if multi-class | int64 | [1] | class_id |
| `text` (可选)  | str      | lowercase, tokenize(max_len=128) | int64 | [seq_len] | NLP 模式 |

---

## 2. Batch 规范 (Batch Specification)
- **batch 输入字典 (dict[str, Tensor])**
  - `images`: torch.FloatTensor `[B, C, H, W]`
  - `labels`: torch.LongTensor `[B]`
  - `attention_mask` (可选): torch.LongTensor `[B, seq_len]`
- **dtype 约束**
  - 所有 float 张量 → `torch.float32`
  - 所有整数类张量 → `torch.int64`
- **padding 规则**
  - 图像 → resize/crop 到统一 H×W
  - 文本 → pad 到最大 seq_len

---

## 3. 模型接口 (Model Interface)
```python
forward(
    images: torch.FloatTensor[B, C, H, W],
    labels: Optional[torch.LongTensor[B]] = None,
    **kwargs
) -> Dict[str, torch.Tensor]
````

* **返回字典**

  * `logits`: torch.FloatTensor `[B, num_classes]`
  * `loss`: torch.FloatTensor `[]` (标量) — 如果传入 labels
* **注意**

  * `forward` 不做数据加载，假设输入已按批处理好
  * 不在 `forward` 内做优化器 step 或梯度清零

---

## 4. 损失与指标接口 (Loss & Metrics)

* **损失函数**

```python
loss_fn(logits: torch.FloatTensor[B, K], labels: torch.LongTensor[B]) -> torch.FloatTensor[]
```

* 默认: CrossEntropyLoss

* 方向: 越小越好 ↓

* **评估指标**

```python
metric_fn(logits: torch.FloatTensor[B, K], labels: torch.LongTensor[B]) -> float
```

* 默认: Accuracy, ↑ 越大越好
* 输出范围: \[0, 1]

---

## 5. 种子与数据切分 (Seed & Split)

* 随机种子: `[1337, 2025, 42]`
* 划分比例: `train/val/test = 80/10/10`
* 划分方式: stratified by `label`

---

## 6. 异常处理 (Error Handling)

| 异常类型    | 处理策略          | 日志级别     |
| ------- | ------------- | -------- |
| 缺失值 NaN | 样本丢弃          | WARN     |
| 图像无法读取  | 样本丢弃，记录路径     | ERROR    |
| 标签越界    | 抛出 ValueError | CRITICAL |

---

## 7. 兼容性声明 (Compatibility Notes)

* 所有接口需保持向后兼容，除非 **PRD.md** 明确需求变更
* 模型与数据模块的接口变动必须在评审后更新本文件
* 与 **TECH\_SPEC.md** 一致，否则以 CONTRACTS.md 为准

---

## 8. 版本与变更记录 (Version & Change Log)

| 版本   | 日期         | 修改内容         | 修改人 |
| ---- | ---------- | ------------ | --- |
| v1.0 | YYYY-MM-DD | 初版           | XXX |
| v1.1 | YYYY-MM-DD | 扩展支持 text 输入 | XXX |

