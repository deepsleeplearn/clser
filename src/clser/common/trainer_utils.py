#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : trainer_utils.py

import os
from typing import Optional
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint
from clser.common.distributed import print_rank0


def resolve_checkpoint(train_args) -> Optional[str]:
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is not None:
            print_rank0(f"发现checkpoint: {last_checkpoint}")

    if train_args.resume_from_checkpoint is not None:
        print_rank0(f"从指定checkpoint恢复: {train_args.resume_from_checkpoint}")
        return train_args.resume_from_checkpoint
    if last_checkpoint is not None:
        print_rank0(f"从最新checkpoint恢复: {last_checkpoint}")
    return last_checkpoint


def save_train_result(trainer: Trainer, train_result, num_train_samples: int):
    trainer.save_model()
    metrics = train_result.metrics
    metrics["train_samples"] = num_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    return metrics


def final_evaluate(trainer: Trainer, eval_dataset, prefix: str = "final_eval"):
    if not trainer.args.do_eval:
        print_rank0(f"跳过最终评估: do_eval=False")
        return None

    metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=prefix)
    metrics[f"{prefix}_samples"] = len(eval_dataset)
    trainer.log_metrics(prefix, metrics)
    trainer.save_metrics(prefix, metrics)
    print_rank0(f"最终评估结果: {metrics}")
    return metrics


def banner(text: str):
    print_rank0("=" * 50)
    print_rank0(text)
    print_rank0("=" * 50)
