# 文本分类

基于 Qwen3 序列分类头的训练脚本，同时支持单标签和多标签分类。代码在 `src/clser/tasks/classification/`。

## 输入数据说明
### 两种模式
#### 对话模式
输入数据为对话形式
##### 格式1
|      键名    |    类型 |  说明   | 示例 |
|:-----------:|:----------:|:-------:|:-----:|
|system|str|系统提示词|你是一个实用小助手|
|conversations|list[str]|交替的以"用户"、"客服"开头的对话内容|["用户: 你好", "客服: 您好，很高兴为您服务，请问是冰箱需要维修吗？", "用户: 是的，冰箱不制冷"]|
|labels|list[str]|文本对应的标签|["抽取品牌品类", "诉求分类"]|

##### 格式2
|      键名    |    类型 |  说明   | 示例 |
|:-----------:|:----------:|:-------:|:-----:|
|system|str|系统提示词|你是一个实用小助手|
|conversations|dict[str, Any]|交替的以"用户"、"客服"开头的对话内容，对话格式为有键role、content的dict|[{"role": "user", "content": 你好"}, {"role": "assistant", "content": "您好，很高兴为您服务，请问是冰箱需要维修吗？"}, {"role": "user", "content": "是的，冰箱不制冷"}]|
|labels|list[str]|文本对应的标签|["抽取品牌品类", "诉求分类"]|

#### 文本模式
##### 格式3
|      键名    |    类型 |  说明   | 示例 |
|:-----------:|:----------:|:-------:|:-----:|
|system|str|系统提示词|你是一个实用小助手|
|text|str|主体文本|"用户：你好。\n客服：您好，很高兴为您服务，请问是冰箱需要维修吗？\n用户：是的，冰箱不制冷"|
|labels|list[str]|文本对应的标签|["抽取品牌品类", "诉求分类"]|

##### 格式4
|      键名    |    类型 |  说明   | 示例 |
|:-----------:|:----------:|:-------:|:-----:|
|system|str|系统提示词|你是一个实用小助手|
|text|str|主体文本|["用户：你好。", "客服：您好，很高兴为您服务，请问是冰箱需要维修吗？", "用户：是的，冰箱不制冷"]|
|labels|list[str]|文本对应的标签|["抽取品牌品类", "诉求分类"]|

### 说明
1. 标签名可以不强制labels，但是要在 yaml 中设置 `label_key` 与数据中一致；
2. 格式 1 中，system、conversations使用换行符"\n"进行join；而格式 4 中，system、text直接使用""进行join；

## 配置
- `configs/classification/multi_label.yaml`：多标签分类，`problem_type: multi_label_classification`
- `configs/classification/single_label.yaml`：单标签分类，`problem_type: single_label_classification`
- `configs/classification/hp.yaml`：超参搜索空间，`use_hyperparameter_search: true` 时通过 `hyperparameter_config_path` 指定

## 运行
```shell
bash scripts/run.sh classification configs/classification/multi_label.yaml
```
