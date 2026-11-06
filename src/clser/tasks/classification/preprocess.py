#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Dict, List, Tuple

import numpy as np


def build_main_text(example: Dict[str, Any], data_args: Any) -> str:
    if data_args.label_key not in example:
        raise ValueError(f"{data_args.label_key} 不在 {example.keys()} 中")

    if "system" in example:
        system = example["system"]
    elif data_args.system_prompt:
        system = data_args.system_prompt
    else:
        raise ValueError("You must provide system prompt.")

    main_text = system

    if "conversations" in example and "text" not in example:
        conversations = example["conversations"]
        if not isinstance(conversations, list):
            raise ValueError("conversations must be a list")
        if not conversations:
            raise ValueError("conversations must not be empty")

        if isinstance(conversations[0], str):
            main_text += "\n".join(conversations)
        elif isinstance(conversations[0], dict):
            for idx, conv in enumerate(conversations):
                if "role" not in conv or "content" not in conv:
                    raise ValueError(
                        "conversations must be a list of dicts with 'role' and 'content' keys"
                    )
                role, content = conv["role"], conv["content"]
                if role not in ["user", "assistant"]:
                    raise ValueError("role must be 'user' or 'assistant'")
                convert_role = "用户" if role == "user" else "客服"
                sep = "\n" if idx != len(conversations) - 1 else ""
                main_text += f"{convert_role}: {content}{sep}"
        else:
            raise ValueError("Elements' type in conversations must be string or dict.")
    elif "text" in example and "conversations" not in example:
        text = example["text"]
        if isinstance(text, str):
            main_text += text
        elif isinstance(text, list):
            main_text += "".join(text)
        else:
            raise ValueError(
                f"Input {text} is not valid. Should be a string, or a list of strings."
            )
    else:
        raise ValueError("input must be a dictionary with a key 'text' or 'conversations'")

    return main_text


def normalize_label_input(raw_label: Any) -> List[str]:
    if isinstance(raw_label, list):
        return raw_label
    if isinstance(raw_label, str):
        return [raw_label]
    raise ValueError("label_key must be list or str")


def build_empty_label(train_args: Any, mlb: Any) -> Any:
    if train_args.problem_type == "multi_label_classification":
        return np.zeros(len(mlb.classes_), dtype=np.float32).tolist()
    if train_args.problem_type == "single_label_classification":
        return 0
    raise ValueError(f"Unknown problem type: {train_args.problem_type}")


def build_label(raw_label: Any, train_args: Any, mlb: Any) -> Any:
    hot_encode = mlb.transform([normalize_label_input(raw_label)])[0]

    if train_args.problem_type == "multi_label_classification":
        return hot_encode.astype(np.float32).tolist()
    if train_args.problem_type == "single_label_classification":
        return int(np.argmax(hot_encode))
    raise ValueError(f"Unknown problem type: {train_args.problem_type}")


def build_preprocess_functions(
    data_args: Any,
    train_args: Any,
    tokenizer: Any,
    mlb: Any,
    logger: Any,
) -> Tuple[Any, Any]:
    empty_label = build_empty_label(train_args, mlb)

    def preprocess_func(example: Dict[str, Any]) -> Dict[str, Any]:
        try:
            main_text = build_main_text(example, data_args)
            result = tokenizer(
                main_text,
                padding=False,
                truncation=False,
                return_tensors=None,
            )
            result["skip"] = len(result["input_ids"]) > data_args.max_length_threshold
            result["label"] = build_label(example[data_args.label_key], train_args, mlb)
            return result
        except Exception as e:
            logger.warning(f"预处理样本时出错: {e}, 跳过该样本")
            return {"skip": True, "label": empty_label}

    def preprocess_func_batched(examples: Dict[str, List[Any]]) -> Dict[str, Any]:
        if data_args.label_key not in examples:
            raise ValueError(f"{data_args.label_key} 不在 {examples.keys()} 中")

        batch_size = len(examples[data_args.label_key])
        main_texts: List[str] = []
        skip_flags = [False] * batch_size
        label_inputs: List[List[str]] = []

        for i in range(batch_size):
            try:
                example = {k: v[i] for k, v in examples.items()}
                main_texts.append(build_main_text(example, data_args))
            except Exception as e:
                logger.warning(f"第 {i} 个样本文本构建失败: {e}, 跳过")
                main_texts.append("")
                skip_flags[i] = True

            try:
                label_inputs.append(normalize_label_input(examples[data_args.label_key][i]))
            except Exception as e:
                logger.warning(f"第 {i} 个样本标签处理失败: {e}, 跳过")
                label_inputs.append([])
                skip_flags[i] = True

        tokenized = tokenizer(
            main_texts,
            padding=False,
            truncation=False,
            return_tensors=None,
        )

        for i, input_ids in enumerate(tokenized["input_ids"]):
            if len(input_ids) > data_args.max_length_threshold:
                skip_flags[i] = True

        hot_encoded = mlb.transform(label_inputs)
        if train_args.problem_type == "multi_label_classification":
            labels = hot_encoded.astype(np.float32).tolist()
        elif train_args.problem_type == "single_label_classification":
            labels = np.argmax(hot_encoded, axis=1).astype(int).tolist()
        else:
            raise ValueError(f"Unknown problem type: {train_args.problem_type}")

        tokenized["label"] = labels
        tokenized["skip"] = skip_flags
        return tokenized

    return preprocess_func, preprocess_func_batched
