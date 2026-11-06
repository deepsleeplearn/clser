#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-01-22
# @Author : GJason
# @File : preprocess.py

import re
from typing import Any, Dict, List, Optional, Tuple

IGNORE_INDEX = -100
_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)


def _normalize_text(value: Any, name: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if not all(isinstance(item, str) for item in value):
            raise ValueError(f"{name} must be a string or a list of strings")
        return "\n".join(value)
    raise ValueError(f"{name} must be a string or a list of strings, but got {type(value).__name__}")


def build_prompt(example: Dict[str, Any], data_args: Any) -> str:
    if data_args.source_key not in example:
        raise ValueError(f"{data_args.source_key} 不在 {list(example.keys())} 中")

    source = _normalize_text(example[data_args.source_key], data_args.source_key)
    if not source.strip():
        raise ValueError(f"{data_args.source_key} must not be empty")

    raw_instruction = example.get(data_args.instruction_key)
    if raw_instruction is None or (isinstance(raw_instruction, str) and not raw_instruction.strip()):
        instruction = data_args.default_instruction
    else:
        instruction = _normalize_text(raw_instruction, data_args.instruction_key)

    prompt = data_args.prompt_template.format(instruction=instruction, input=source)
    return prompt.strip()


def build_target(example: Dict[str, Any], data_args: Any) -> str:
    if data_args.target_key not in example:
        raise ValueError(f"{data_args.target_key} 不在 {list(example.keys())} 中")
    target = _normalize_text(example[data_args.target_key], data_args.target_key).strip()
    if not target:
        raise ValueError(f"{data_args.target_key} must not be empty")
    return target


def resolve_system(example: Dict[str, Any], data_args: Any) -> str:
    system = example.get("system")
    if system is None:
        system = data_args.system_prompt
    if not isinstance(system, str):
        raise ValueError("system must be a string")
    return system.strip()


def build_messages(example: Dict[str, Any], data_args: Any) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    system = resolve_system(example, data_args)
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": build_prompt(example, data_args)})
    return messages


def _chat_template_kwargs(data_args: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if getattr(data_args, "enable_thinking", None) is not None:
        kwargs["enable_thinking"] = data_args.enable_thinking
    return kwargs


def has_chat_template(tokenizer: Any) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def uses_chat_template(data_args: Any, tokenizer: Any) -> bool:
    return bool(data_args.use_chat_template) and has_chat_template(tokenizer)


def causal_add_special_tokens(data_args: Any, tokenizer: Any) -> bool:
    return not uses_chat_template(data_args, tokenizer)


def build_causal_prompt_text(example: Dict[str, Any], data_args: Any, tokenizer: Any) -> str:
    messages = build_messages(example, data_args)
    if uses_chat_template(data_args, tokenizer):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **_chat_template_kwargs(data_args),
        )
    parts = [m["content"] for m in messages]
    return "\n".join(parts) + "\n"


def build_seq2seq_source_text(example: Dict[str, Any], data_args: Any) -> str:
    system = resolve_system(example, data_args)
    prompt = build_prompt(example, data_args)
    body = f"{system}\n{prompt}" if system else prompt
    return f"{data_args.source_prefix}{body}"


def build_eval_prompt_text(example: Dict[str, Any], data_args: Any, model_type: str, tokenizer: Any) -> str:
    if model_type == "causal_lm":
        return build_causal_prompt_text(example, data_args, tokenizer)
    if model_type == "seq2seq_lm":
        return build_seq2seq_source_text(example, data_args)
    raise ValueError(f"Unknown model_type: {model_type}")


def clean_generation(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


def encode_causal_example(
    example: Dict[str, Any],
    data_args: Any,
    tokenizer: Any,
) -> Tuple[List[int], List[int], List[int], bool]:
    prompt_text = build_causal_prompt_text(example, data_args, tokenizer)
    target = build_target(example, data_args)

    prompt_ids = tokenizer(
        prompt_text, add_special_tokens=causal_add_special_tokens(data_args, tokenizer)
    )["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        target_ids = target_ids + [eos_id]

    skip = (
        len(prompt_ids) > data_args.max_source_length
        or len(target_ids) > data_args.max_target_length
        or len(prompt_ids) + len(target_ids) > data_args.max_length_threshold
    )

    input_ids = prompt_ids + target_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, labels, skip


def encode_seq2seq_example(
    example: Dict[str, Any],
    data_args: Any,
    tokenizer: Any,
) -> Tuple[List[int], List[int], List[int], bool]:
    source_text = build_seq2seq_source_text(example, data_args)
    target = build_target(example, data_args)

    model_inputs = tokenizer(source_text, padding=False, truncation=False)
    labels = tokenizer(text_target=target, padding=False, truncation=False)["input_ids"]

    skip = (
        len(model_inputs["input_ids"]) > data_args.max_source_length
        or len(labels) > data_args.max_target_length
    )
    return model_inputs["input_ids"], model_inputs["attention_mask"], labels, skip


def build_preprocess_functions(
    data_args: Any,
    model_type: str,
    tokenizer: Any,
    logger: Any,
) -> Tuple[Any, Any]:
    if model_type == "causal_lm":
        encode_example = encode_causal_example
    elif model_type == "seq2seq_lm":
        encode_example = encode_seq2seq_example
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    empty_result = {
        "input_ids": [pad_id],
        "attention_mask": [1],
        "labels": [IGNORE_INDEX],
        "skip": True,
    }

    def preprocess_func(example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            input_ids, attention_mask, labels, skip = encode_example(example, data_args, tokenizer)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "skip": skip,
            }
        except Exception as e:
            logger.warning(f"预处理样本时出错: {e}, 跳过该样本")
            return dict(empty_result)

    def preprocess_func_batched(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        if data_args.target_key not in examples:
            raise ValueError(f"{data_args.target_key} 不在 {list(examples.keys())} 中")

        batch_size = len(examples[data_args.target_key])
        result: Dict[str, List[Any]] = {"input_ids": [], "attention_mask": [], "labels": [], "skip": []}

        for i in range(batch_size):
            example = {k: v[i] for k, v in examples.items()}
            try:
                input_ids, attention_mask, labels, skip = encode_example(example, data_args, tokenizer)
            except Exception as e:
                logger.warning(f"第 {i} 个样本预处理失败: {e}, 跳过")
                input_ids, attention_mask, labels, skip = (
                    empty_result["input_ids"],
                    empty_result["attention_mask"],
                    empty_result["labels"],
                    True,
                )
            result["input_ids"].append(input_ids)
            result["attention_mask"].append(attention_mask)
            result["labels"].append(labels)
            result["skip"].append(skip)

        return result

    return preprocess_func, preprocess_func_batched


def tokenize_for_metric(text: str, mode: str, tokenizer: Optional[Any] = None) -> List[str]:
    text = text.strip()
    if mode == "char":
        return [ch for ch in text if not ch.isspace()]
    if mode == "whitespace":
        return text.split()
    if mode == "model":
        if tokenizer is None:
            raise ValueError("metric_tokenize='model' requires a tokenizer")
        return tokenizer.tokenize(text)
    raise ValueError(f"Unknown metric_tokenize: {mode}")
