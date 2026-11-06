#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : preprocess.py

import random
from typing import Any, Dict, List, Tuple

DEFAULT_LLM_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
    'Note that the answer can only be "yes" or "no".'
)


def _as_list(value: Any, name: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError(f"{name} must be a string or a list of strings")


def _truncate(tokenizer: Any, text: str, max_tokens: int) -> str:
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=max_tokens)["input_ids"]
    return tokenizer.decode(ids, skip_special_tokens=True)


def resolve_instruction(example: Dict[str, Any], data_args: Any) -> str:
    instruction = example.get(data_args.instruction_key)
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = data_args.default_instruction
    return instruction.strip()


def sample_group(example: Dict[str, Any], data_args: Any, group_size: int) -> Tuple[str, List[str]]:
    for key in [data_args.query_key, data_args.pos_key, data_args.neg_key]:
        if key not in example:
            raise ValueError(f"{key} 不在 {list(example.keys())} 中")

    query = example[data_args.query_key]
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{data_args.query_key} must be a non-empty string")

    pos = [p for p in _as_list(example[data_args.pos_key], data_args.pos_key) if p.strip()]
    neg = [n for n in _as_list(example[data_args.neg_key], data_args.neg_key) if n.strip()]
    if not pos:
        raise ValueError(f"{data_args.pos_key} must contain at least one passage")
    if not neg:
        raise ValueError(f"{data_args.neg_key} must contain at least one passage")

    rng = random.Random(f"{data_args.neg_sample_seed}-{query}")
    positive = rng.choice(pos)
    num_neg = group_size - 1
    if len(neg) >= num_neg:
        negatives = rng.sample(neg, num_neg)
    else:
        negatives = list(neg)
        while len(negatives) < num_neg:
            negatives.append(rng.choice(neg))
    return query.strip(), [positive] + negatives


def build_llm_prompt(instruction: str, query: str, passage: str, data_args: Any, tokenizer: Any) -> str:
    system = data_args.system_prompt.strip() or DEFAULT_LLM_SYSTEM
    user = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {passage}"
    if getattr(tokenizer, "chat_template", None) is None:
        return f"{system}\n{user}\n"

    kwargs = {}
    if data_args.enable_thinking is not None:
        kwargs["enable_thinking"] = data_args.enable_thinking
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def encode_cross_encoder_group(
    example: Dict[str, Any], data_args: Any, tokenizer: Any, group_size: int
) -> Tuple[List[List[int]], List[List[int]], bool]:
    query, passages = sample_group(example, data_args, group_size)
    query = _truncate(tokenizer, query, data_args.max_query_length)
    passages = [_truncate(tokenizer, p, data_args.max_passage_length) for p in passages]
    encoded = tokenizer(
        [query] * len(passages),
        passages,
        truncation="longest_first",
        max_length=data_args.max_length_threshold,
        padding=False,
    )
    return encoded["input_ids"], encoded["attention_mask"], False


def encode_llm_group(
    example: Dict[str, Any], data_args: Any, tokenizer: Any, group_size: int
) -> Tuple[List[List[int]], List[List[int]], bool]:
    query, passages = sample_group(example, data_args, group_size)
    instruction = resolve_instruction(example, data_args)
    query = _truncate(tokenizer, query, data_args.max_query_length)
    passages = [_truncate(tokenizer, p, data_args.max_passage_length) for p in passages]
    prompts = [build_llm_prompt(instruction, query, p, data_args, tokenizer) for p in passages]
    encoded = tokenizer(prompts, add_special_tokens=False, truncation=False, padding=False)
    skip = any(len(ids) > data_args.max_length_threshold for ids in encoded["input_ids"])
    return encoded["input_ids"], encoded["attention_mask"], skip


def build_preprocess_functions(
    data_args: Any,
    model_type: str,
    tokenizer: Any,
    logger: Any,
    group_size: int,
) -> Tuple[Any, Any]:
    if model_type == "cross_encoder":
        encode_group = encode_cross_encoder_group
    elif model_type == "llm":
        encode_group = encode_llm_group
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    empty_result = {
        "input_ids": [[pad_id]] * group_size,
        "attention_mask": [[1]] * group_size,
        "skip": True,
    }

    def preprocess_func(example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_ids, attention_mask, skip = encode_group(example, data_args, tokenizer, group_size)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "skip": skip}
        except Exception as e:
            logger.warning(f"预处理样本时出错: {e}, 跳过该样本")
            return dict(empty_result)

    def preprocess_func_batched(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        if data_args.query_key not in examples:
            raise ValueError(f"{data_args.query_key} 不在 {list(examples.keys())} 中")

        batch_size = len(examples[data_args.query_key])
        result: Dict[str, List[Any]] = {"input_ids": [], "attention_mask": [], "skip": []}
        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            try:
                input_ids, attention_mask, skip = encode_group(example, data_args, tokenizer, group_size)
            except Exception as e:
                logger.warning(f"第 {i} 个样本预处理失败: {e}, 跳过")
                input_ids, attention_mask, skip = empty_result["input_ids"], empty_result["attention_mask"], True
            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            result["skip"].append(skip)
        return result

    return preprocess_func, preprocess_func_batched
