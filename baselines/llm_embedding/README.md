# LLM Embedding Baseline（第一阶段）

独立于 TMGCN 核心网络。不修改 `models/` 的 forward，不覆盖原始数据。

## 任务与数据（来自现有代码，不是假设）

- 数据集：Zero-shot Conversational Stance Detection（CSD / ZS-CSD）。
- 原始 JSON 每条记录是**一段完整会话**（`sentences` + `speakers` + `target`）。
- `all_labels[i]` 是第 i 句立场：`0=favor, 1=against, 2=neutral`（`<0` 忽略）。
- `label` 是文档级标签。TMGCN 官方评价（`metric_on_last_sentence=true`）按 **每个 `doc_id` 的最后一句** 计分。
- TMGCN 训练单元是 **target 图**（同一 target 下所有会话的句子做节点），不是会话级分类。
- train / dev / test **target 不相交**（零样本）。

因此：

| 粒度 | 做法 | 是否合法 |
|---|---|---|
| 会话向量 → 最后一句标签 | `--unit last_sentence`（默认） | 合法，对齐官方指标 |
| 一个会话向量复制给所有句子 | 禁止 | 标签粒度不匹配 |
| 每句一个前缀向量 | `--unit prefix` | 合法的话语级表示，更贵 |

## 提取

```bash
python baselines/llm_embedding/extract_conversation_embeddings.py \
  --data_dir /path/to/raw_with_global_ids \
  --output_dir /path/to/cache \
  --model Qwen/Qwen3-Embedding-0.6B \
  --unit last_sentence \
  --batch_size 1 --max_length 2048 --dtype bf16 --device cuda:0
```

超长会话：`--overflow error|truncate|chunk`。`error` 会打印长度统计后退出，不静默丢数据。`truncate` 保留尾部（最后一句）。

## 分类

```bash
python baselines/llm_embedding/train_classifier.py \
  --cache_dir /path/to/cache \
  --output_dir /path/to/result \
  --classifier mlp --epochs 20 --seed 1234
```

指标与 TMGCN 相同：mixed Macro F1 + Favor / Against / Neutral，并分 claim(c) / noun-phrase(n)。
