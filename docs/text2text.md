# 文本到文本生成（摘要 / 总结 / 翻译）

代码在 `src/clser/tasks/text2text/`。数据参数类继承分类任务的 `MultiClassificationDataArguments`，公共字段含义一致。

## 支持的模型类型
|  model_type   |  模型类  |  适用模型  |  说明  |
|:-------------:|:-------:|:---------:|:------|
| `causal_lm`   | `AutoModelForCausalLM`  | Qwen3 / Llama 等解码器模型 | 用 chat template 拼 prompt，只对回答部分计算 loss（prompt 位置的 label 为 -100） |
| `seq2seq_lm`  | `AutoModelForSeq2SeqLM` | T5 / mT5 / BART 等编码器-解码器模型 | 源文本进编码器，目标文本做 labels，可加 `source_prefix` |

## 输入数据说明
jsonl 每行一个样本，字段名可通过 `instruction_key` / `source_key` / `target_key` 修改，默认如下：

|      键名    |    类型 |  是否必填 |  说明   | 示例 |
|:-----------:|:----------:|:-------:|:-------:|:-----:|
|system|str|否|系统提示词，缺省时使用 yaml 中的 `system_prompt`（为空则不加 system 消息）|你是一个实用小助手|
|instruction|str|否|任务指令，缺省时使用 yaml 中的 `default_instruction`|请总结下面的对话|
|input|str 或 list[str]|是|源文本；list 会用换行符 join|"用户：冰箱不制冷。\n客服：好的，帮您预约维修。"|
|output|str|是|目标文本|"用户反馈冰箱不制冷，客服已预约维修。"|

prompt 由 `prompt_template`（默认 `"{instruction}\n{input}"`）拼接而成。超过 `max_source_length` / `max_target_length` / `max_length_threshold` 的样本会被跳过而不是截断。`test_file_path` 必须出现（可以为 `null`），`label_key` / `*_remove_columns` 不生效，原始文本列在 tokenize 后自动删除。

## 评估
- 训练过程中按 `eval_steps` 计算验证集 loss。`seq2seq_lm` 可开启 `predict_with_generate` 在训练中直接算生成指标；`causal_lm` 的训练样本已包含答案，该选项会被强制关闭。
- 训练结束后，若 `do_generation_eval: true`，rank 0 会从 `output_dir` 重新加载模型，对验证集（以及 `test_file_path`）做生成，计算 ROUGE-1/2/L 与 BLEU，结果写入 `output_dir/generation_eval_results.json`，逐条结果写入 `output_dir/generation_<split>_predictions.jsonl`。
- 指标在 `metrics.py` 中本地实现，`metric_tokenize` 控制分词粒度：中文用 `char`，英文用 `whitespace`，也可用 `model` 走模型分词。

## 配置与运行
- `configs/text2text/causal_lm.yaml`：Qwen3 对话总结
- `configs/text2text/seq2seq.yaml`：mT5 翻译
```shell
bash scripts/run.sh text2text configs/text2text/causal_lm.yaml
```
