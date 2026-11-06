#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : data.py

from typing import Iterable, Optional
from datasets import load_dataset, DatasetDict
from clser.common.distributed import print_rank0
from clser.utils.logger import get_logger

logger = get_logger()


def load_json_splits(train_file_path: str, valid_file_path: str) -> DatasetDict:
    try:
        train_dataset = load_dataset("json", data_files=train_file_path)
        valid_dataset = load_dataset("json", data_files=valid_file_path)
    except Exception as e:
        logger.error(f"加载数据集失败: {e}")
        raise

    return DatasetDict({
        "train": train_dataset["train"],
        "valid": valid_dataset["train"]
    })


def require_columns(dataset_dict: DatasetDict, keys: Iterable[str]):
    for split in dataset_dict:
        columns = dataset_dict[split].column_names
        for key in keys:
            if key not in columns:
                raise ValueError(f"{split} 数据缺少必需字段 {key!r}，现有字段: {columns}")


def filter_skipped(dataset, split_name: str, num_proc: int, keep_columns: Optional[Iterable[str]] = None):
    original_size = len(dataset)
    dataset = dataset.filter(
        function=lambda x: not x.get("skip", False),
        num_proc=num_proc
    )
    if keep_columns is None:
        extra_columns = ["skip"]
    else:
        keep_columns = set(keep_columns)
        extra_columns = [c for c in dataset.column_names if c not in keep_columns]
    dataset = dataset.remove_columns(extra_columns)
    print_rank0(
        f"{split_name}过滤: {original_size} -> {len(dataset)} "
        f"(跳过 {original_size - len(dataset)} 个超长或无效样本); 已删除列: {extra_columns}"
    )
    return dataset
