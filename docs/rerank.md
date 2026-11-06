# 重排序模型训练（Reranker）

代码在 `src/clser/tasks/rerank/`。数据参数类继承分类任务的 `MultiClassificationDataArguments`。

## 支持的模型类型
|  model_type   |  模型类  |  适用模型  |  打分方式  |
|:-------------:|:-------:|:---------:|:------|
| `cross_encoder` | `AutoModelForSequenceClassification`（num_labels=1） | bge-reranker-base / large / v2-m3、bce-reranker 等 BERT 系 | query 与 passage 拼成一个序列，取分类头输出 |
| `llm` | `AutoModelForCausalLM` | Qwen3-Reranker-0.6B / 4B / 8B，或任意 Qwen3 / Llama 底座 | 按 Qwen3-Reranker 的 Instruct / Query / Document 模板拼 prompt，分数为最后一个位置 `logit(yes) - logit(no)` |

## 输入数据说明
jsonl 每行一个 query，字段名可通过 `query_key` / `pos_key` / `neg_key` / `instruction_key` 修改：

|      键名    |    类型 |  是否必填 |  说明   |
|:-----------:|:----------:|:-------:|:-------:|
|query|str|是|查询文本|
|pos|str 或 list[str]|是|正例 passage，训练时每条随机取一个|
|neg|str 或 list[str]|是|负例 passage，建议用 embedding 召回得到的难负例；不足 `group_size - 1` 条时重复采样|
|prompt|str|否|任务指令，仅 `llm` 使用；缺省用 yaml 的 `default_instruction`|

示例：
```json
{"query": "冰箱不制冷怎么办", "pos": ["冰箱不制冷常见原因有压缩机故障、制冷剂泄漏……"], "neg": ["冰箱噪音大的解决方法……", "冰箱门封条老化更换教程……", "洗衣机排水不畅的处理方法……"]}
```

每个 query 组成 `train_group_size` 条样本（1 正例 + N 负例，正例固定在第 0 位），采样种子由 `neg_sample_seed` 与 query 共同决定，多进程预处理下结果可复现。query 和 passage 分别按 `max_query_length` / `max_passage_length` 截断，总长受 `max_length_threshold` 约束（`cross_encoder` 截断，`llm` 跳过）。

## 损失与评估
- `loss_type: listwise`（默认）：组内 softmax 交叉熵，可用 `temperature` 调节；`pointwise`：每对做二元交叉熵。
- 验证集按 `eval_group_size` 组织，指标为正例在组内的排名统计：`accuracy`（排第一的比例）、`mrr`、`ndcg`。
- 训练结束会在 `output_dir` 写入 `rerank_meta.json`，记录模型类型、模板、yes/no token 等推理打分所需的信息。

## 配置与运行
- `configs/rerank/cross_encoder.yaml`：bge-reranker-v2-m3
- `configs/rerank/llm.yaml`：Qwen3-Reranker-0.6B
```shell
bash scripts/run.sh rerank configs/rerank/cross_encoder.yaml
```
