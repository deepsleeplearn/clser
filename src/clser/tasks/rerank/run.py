#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-06-02
# @Author : GJason
# @File : run.py

import torch
import os
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    EvalPrediction,
    set_seed
)
from clser.common.distributed import setup_distributed, print_rank0, cleanup_distributed
from clser.common.args import parse_yaml_args
from clser.common.data import load_json_splits, require_columns, filter_skipped
from clser.common.trainer_utils import resolve_checkpoint, save_train_result, final_evaluate, banner
from clser.tasks.rerank.args import (
    RerankDataArguments,
    RerankModelArguments,
    RerankTrainArguments,
)
from clser.tasks.rerank.preprocess import build_preprocess_functions, DEFAULT_LLM_SYSTEM
from clser.tasks.rerank.trainer import RerankCollator, RerankTrainer
from clser.utils.logger import get_logger

logger = get_logger()

MODEL_COLUMNS = {"input_ids", "attention_mask"}


def load_tokenizer(model_args: "RerankModelArguments"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; cannot pad batches.")
        tokenizer.pad_token = tokenizer.eos_token
        print_rank0(f"tokenizer 没有 pad_token，已设置为 eos_token: {tokenizer.eos_token}")
    if model_args.model_type == "llm":
        tokenizer.padding_side = "left"
    return tokenizer


def get_single_token_id(tokenizer, token: str) -> int:
    ids = tokenizer.encode(token, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{token!r} 必须是单个 token，但被切成了 {ids}")
    return ids[0]


def get_dataset(data_args: "RerankDataArguments",
                train_args: "RerankTrainArguments",
                model_args: "RerankModelArguments",
                tokenizer: "AutoTokenizer"
                ):

    with train_args.main_process_first(desc="pre-process dataset", local=True):

        format_dataset = load_json_splits(data_args.train_file_path, data_args.valid_file_path)
        require_columns(format_dataset, [data_args.query_key, data_args.pos_key, data_args.neg_key])

        def map_split(split, group_size, desc):
            preprocess_func, preprocess_func_batched = build_preprocess_functions(
                data_args=data_args,
                model_type=model_args.model_type,
                tokenizer=tokenizer,
                logger=logger,
                group_size=group_size,
            )
            map_kwargs = {
                "function": preprocess_func_batched if data_args.use_batched_preprocess else preprocess_func,
                "num_proc": data_args.num_processing,
                "desc": desc,
            }
            if data_args.use_batched_preprocess:
                map_kwargs["batched"] = True
                map_kwargs["batch_size"] = data_args.preprocess_batch_size
            return format_dataset[split].map(**map_kwargs)

        print_rank0(
            f"训练组大小: {data_args.train_group_size} (1 正例 + {data_args.train_group_size - 1} 负例), "
            f"验证组大小: {data_args.eval_group_size}"
        )

        train_datasets = map_split("train", data_args.train_group_size, "Running tokenizer on train dataset")
        if data_args.shuffle:
            train_datasets = train_datasets.shuffle(seed=data_args.shuffle_seed)
        train_datasets = filter_skipped(train_datasets, "训练集", data_args.num_processing, MODEL_COLUMNS)

        valid_datasets = map_split("valid", data_args.eval_group_size, "Running tokenizer on valid dataset")
        valid_datasets = filter_skipped(valid_datasets, "验证集", data_args.num_processing, MODEL_COLUMNS)

        if len(train_datasets) == 0:
            raise ValueError("训练集过滤后为空，请检查数据格式或放宽长度阈值。")

        lengths = [len(ids) for group in train_datasets["input_ids"] for ids in group]
        print_rank0(
            f"训练集 query 数: {len(train_datasets)}, 序列长度 min/mean/max: "
            f"{int(np.min(lengths))}/{float(np.mean(lengths)):.1f}/{int(np.max(lengths))}"
        )

        return train_datasets, valid_datasets


def get_model(model_args: "RerankModelArguments",
              train_args: "RerankTrainArguments",
              tokenizer: "AutoTokenizer",
              local_rank: int
              ):

    try:
        if model_args.model_type == "cross_encoder":
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=1,
                trust_remote_code=model_args.trust_remote_code,
            )
            model_cls = AutoModelForSequenceClassification
        else:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                trust_remote_code=model_args.trust_remote_code,
            )
            model_cls = AutoModelForCausalLM

        config._attn_implementation = train_args.attn_implementation
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = tokenizer.pad_token_id
        if hasattr(config, "use_cache"):
            config.use_cache = False

        model = model_cls.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            dtype=torch.bfloat16,
            trust_remote_code=model_args.trust_remote_code,
        )

        if local_rank == 0:
            total_params = model.num_parameters()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print_rank0(f"模型类型: {model_args.model_type} ({model.__class__.__name__})")
            print_rank0(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
            print_rank0(f"模型配置:\n{json.dumps(config.to_dict(), indent=2, ensure_ascii=False, default=str)}")

        model.train()
        return model

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def save_rerank_meta(data_args, model_args, train_args):
    meta = {
        "model_type": model_args.model_type,
        "loss_type": train_args.loss_type,
        "max_query_length": data_args.max_query_length,
        "max_passage_length": data_args.max_passage_length,
        "max_length_threshold": data_args.max_length_threshold,
    }
    if model_args.model_type == "llm":
        meta.update({
            "system_prompt": data_args.system_prompt.strip() or DEFAULT_LLM_SYSTEM,
            "default_instruction": data_args.default_instruction,
            "enable_thinking": data_args.enable_thinking,
            "yes_token": model_args.yes_token,
            "no_token": model_args.no_token,
            "score": "logit(yes_token) - logit(no_token) at the last position",
        })
    meta_path = os.path.join(train_args.output_dir, "rerank_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print_rank0(f"推理所需的打分配置已保存到 {meta_path}")


def run_rerank(yaml_path: str):

    local_rank, rank, world_size = setup_distributed()
    print_rank0(f"Starting rerank training on {world_size} GPU(s)")

    data_args, model_args, train_args = parse_yaml_args(
        yaml_path, (RerankDataArguments, RerankModelArguments, RerankTrainArguments)
    )

    if train_args.seed is not None:
        set_seed(train_args.seed)
        print_rank0(f"Random seed set to {train_args.seed}")

    tokenizer = load_tokenizer(model_args)
    print_rank0(f"Tokenizer加载完成, vocab_size={len(tokenizer)}, padding_side={tokenizer.padding_side}")

    yes_token_id = no_token_id = None
    if model_args.model_type == "llm":
        yes_token_id = get_single_token_id(tokenizer, model_args.yes_token)
        no_token_id = get_single_token_id(tokenizer, model_args.no_token)
        print_rank0(f"llm 打分 token: {model_args.yes_token!r}={yes_token_id}, {model_args.no_token!r}={no_token_id}")

    train_datasets, valid_datasets = get_dataset(
        data_args,
        train_args,
        model_args,
        tokenizer,
    )

    model = get_model(model_args, train_args, tokenizer, local_rank)

    def compute_metrics(p: EvalPrediction):
        scores = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        scores = np.asarray(scores, dtype=np.float32)
        ranks = (scores > scores[:, :1]).sum(axis=1) + 1
        result = {}
        for measure in train_args.measures:
            if measure == "accuracy":
                result[measure] = float((ranks == 1).mean())
            elif measure == "mrr":
                result[measure] = float((1.0 / ranks).mean())
            elif measure == "ndcg":
                result[measure] = float((1.0 / np.log2(ranks + 1)).mean())
            else:
                raise ValueError(f"unknown measure: {measure}")
        return result

    trainer = RerankTrainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=RerankCollator(tokenizer, pad_to_multiple_of=data_args.pad_to_multiple_of),
        processing_class=tokenizer,
        train_dataset=train_datasets,
        eval_dataset=valid_datasets,
        model_type=model_args.model_type,
        loss_type=train_args.loss_type,
        temperature=train_args.temperature,
        train_group_size=data_args.train_group_size,
        eval_group_size=data_args.eval_group_size,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
    )

    checkpoint = resolve_checkpoint(train_args)

    try:
        banner("开始训练")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print_rank0(train_result)

        save_train_result(trainer, train_result, len(train_datasets))
        final_evaluate(trainer, valid_datasets)

        if trainer.is_world_process_zero():
            save_rerank_meta(data_args, model_args, train_args)

        banner("训练完成")

    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise

    finally:
        cleanup_distributed()
