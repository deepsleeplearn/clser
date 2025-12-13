#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-12-12
# @Author : GJason
# @File : run_classification.py

import torch
import os
import sys
import json
import yaml
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from transformers import (
    AutoTokenizer, 
    Qwen3ForSequenceClassification, 
    AutoConfig,
    EvalPrediction,
    Trainer,
    HfArgumentParser,
    default_data_collator,
    set_seed
)
from transformers.trainer_utils import get_last_checkpoint
import evaluate
import torch.distributed as dist
import torch.nn.functional as F
from clser.config.multilabel_task_args import (
    MultiClassificationDataArguments,
    MultiClassificationTrainArguments,
    MultiClassificationModelArguments,
)
from clser.utils.logger import get_logger

logger = get_logger()


def setup_distributed():
    
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))

        torch.cuda.set_device(local_rank)
        
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
                device_id=torch.device(f"cuda:{local_rank}")
            )
            logger.info(f"Initialized process group: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        
        return local_rank, rank, world_size
    else:
        return 0, 0, 1


def print_rank0(info):
    if not dist.is_initialized() or dist.get_rank() == 0:
        logger.info(info)


def get_dataset(
    data_args: "MultiClassificationDataArguments",
    train_args: "MultiClassificationTrainArguments", 
    model_args: "MultiClassificationModelArguments",
    tokenizer: "AutoTokenizer"
):
    
    mlb = MultiLabelBinarizer(classes=model_args.multi_classes)
    mlb.fit([model_args.multi_classes])
    
    with train_args.main_process_first(desc="pre-process dataset", local=True):
        
        try:
            train_dataset = load_dataset("json", data_files=data_args.train_file_path)
            valid_dataset = load_dataset("json", data_files=data_args.valid_file_path)
        except Exception as e:
            logger.error(f"加载数据集失败: {e}")
            raise

        format_dataset = DatasetDict({
            "train": train_dataset["train"],
            "valid": valid_dataset["train"]
        })


        label_set = set()
        if train_args.problem_type == "multi_label_classification":
            for labels in format_dataset["train"][data_args.label_key]:
                for label in labels:
                    label_set.add(label)
            for labels in format_dataset["valid"][data_args.label_key]:
                for label in labels:
                    label_set.add(label)
        elif train_args.problem_type == "single_label_classification":
            for label in format_dataset["train"][data_args.label_key]:
                label_set.add(label)
            for label in format_dataset["valid"][data_args.label_key]:
                label_set.add(label)
        else:
            raise ValueError(f"Unknown problem_type type: {train_args.problem_type}")
                
        for cur_label in list(label_set):
            assert cur_label in model_args.multi_classes, f"{cur_label} 不在 {model_args.multi_classes} 中"  
            
        print_rank0(f"发现 {len(label_set)} 个唯一标签: {label_set}")

        def preprocess_func(example):
            try:
                assert data_args.label_key in example, f"{data_args.label_key} 不在 {example.keys()} 中"
                
                main_text = ""

                if "system" in example:
                    system = example["system"]
                else:
                    system = "你是一个实用小助手,我需要你根据用户和客服的对话数据上下文帮我判断用户所说的最后一句话对应的状态。对话内容如下:\n"
                
                main_text += system

                if "conversations" in example and "text" not in example:
                    conversations = example["conversations"]
                    l = len(conversations)
                    assert isinstance(conversations, list), "conversations must be a list"
                    
                    if isinstance(conversations[0], str):
                        main_text += "\n".join(conversations)
                    elif isinstance(conversations[0], dict):
                        assert "role" in conversations[0] and "content" in conversations[0], \
                            "conversations must be a list of dicts with 'role' and 'content' keys"
                        for idx, conv in enumerate(conversations):
                            role, content = conv["role"], conv["content"]
                            assert role in ["user", "assistant"], "role must be 'user' or 'assistant'"
                            if data_args.task_name == "state_model":
                                assert role == "user" if idx % 2 == 0 else role == "assistant"
                            convert_role = "用户" if role == "user" else "客服"
                            if idx != (l - 1):
                                main_text += f"{convert_role}:{content}\n"
                            else:
                                main_text += f"{convert_role}:{content}"
                    else:
                        raise ValueError(f"Unknown example type: {example}")
                        
                elif "text" in example and "conversations" not in example:
                    text = example["text"]
                    if isinstance(text, str):
                        main_text += text
                    elif isinstance(text, list):
                        main_text += "".join(text)
                    else:
                        raise ValueError(f"Input {text} is not valid. Should be a string, or a list of strings.")
                else:
                    raise ValueError(
                        "input must be a dictionary with a key 'text' or 'conversations'"
                    )
                        
                full_text_tokens = tokenizer.encode(main_text, add_special_tokens=True)

                if len(full_text_tokens) > data_args.max_length_threshold:
                    example["skip"] = True
                else:
                    example["skip"] = False

                result = tokenizer(
                    main_text, 
                    padding="max_length", 
                    max_length=data_args.max_length_threshold,
                    return_tensors='pt'
                )
                
                for k, v in result.items():
                    result[k] = v[0]

                if isinstance(example[data_args.label_key], list):
                    hot_encode = mlb.transform([example[data_args.label_key]])[0]
                elif isinstance(example[data_args.label_key], str):
                    hot_encode = mlb.transform([[example[data_args.label_key]]])[0]
                else:
                    raise ValueError("label_key must be list or str")
                
                if train_args.problem_type == "multi_label_classification":
                    result["label"] = hot_encode
                elif train_args.problem_type == "single_label_classification":
                    result["label"] = np.argmax(hot_encode)
                else:
                    raise ValueError(f"Unknown problem type: {train_args.problem_type}")
                    
                return result
                
            except Exception as e:
                logger.warning(f"预处理样本时出错: {e}, 跳过该样本")
                return {"skip": True}

        train_datasets = format_dataset["train"]
        train_datasets = train_datasets.map(
            function=preprocess_func, 
            num_proc=data_args.num_processing, 
            desc="Running tokenizer on train dataset",
        )
        
        train_datasets = train_datasets.remove_columns(data_args.train_remove_columns)

        if data_args.shuffle:
            train_datasets = train_datasets.shuffle(seed=data_args.shuffle_seed)

        valid_datasets = format_dataset["valid"]
        valid_datasets = valid_datasets.map(
            function=preprocess_func,
            num_proc=data_args.num_processing, 
            desc="Running tokenizer on valid dataset",
            remove_columns=data_args.valid_remove_columns
        )

        original_train_size = len(train_datasets)
        train_datasets = train_datasets.filter(
            function=lambda x: not x.get("skip", False), 
            num_proc=data_args.num_processing
        )
        filtered_train_size = len(train_datasets)
        train_datasets = train_datasets.remove_columns(["skip"])
        print_rank0(f"训练集过滤: {original_train_size} -> {filtered_train_size} (跳过 {original_train_size - filtered_train_size} 个超长样本)")

        original_valid_size = len(valid_datasets)
        valid_datasets = valid_datasets.filter(
            function=lambda x: not x.get("skip", False), 
            num_proc=data_args.num_processing
        )
        filtered_valid_size = len(valid_datasets)
        valid_datasets = valid_datasets.remove_columns(["skip"])
        print_rank0(f"验证集过滤: {original_valid_size} -> {filtered_valid_size} (跳过 {original_valid_size - filtered_valid_size} 个超长样本)")

        train_shape = np.array(train_datasets["label"]).shape
        print_rank0(f"训练集标签维度: {train_shape}") 
        print_rank0(f"训练集input_ids维度: {np.array(train_datasets['input_ids']).shape}")

        return train_datasets, valid_datasets


def get_model(config: "AutoConfig",
              model_args: "MultiClassificationModelArguments",
              local_rank: int
              ):
    
    try:
        model = Qwen3ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path, 
            config=config,
            dtype=torch.bfloat16
        )
        
        if local_rank == 0:
            total_params = model.num_parameters()
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print_rank0(f"模型总参数: {total_params:,}, 可训练参数: {trainable_params:,}")
            print_rank0(f"模型所在设备: {model.device}")
            score_params = sum(p.numel() for name, p in model.named_parameters() if "score" in name)
            print_rank0(f"score 层参数: {score_params:,}")
                    
        model.train()
        return model
        
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def get_args(args_file: str):
    hf_parser = HfArgumentParser((
        MultiClassificationDataArguments, 
        MultiClassificationModelArguments, 
        MultiClassificationTrainArguments
    ))
    with open(args_file, 'r', encoding='utf-8') as f:
        yaml_dict = yaml.safe_load(f)
    data_args, model_args, train_args = hf_parser.parse_dict(yaml_dict)
    return data_args, model_args, train_args


def run_classification():
    
    local_rank, rank, world_size = setup_distributed()
    print_rank0(f"Starting training on {world_size} GPU(s)")
    
    yaml_path = sys.argv[1]
    assert os.path.exists(yaml_path) and os.path.isfile(yaml_path), \
        f"yaml_path does not exist: {yaml_path}"

    data_args, model_args, train_args = get_args(yaml_path)
    
    if train_args.seed is not None:
        set_seed(train_args.seed)
        print_rank0(f"Random seed set to {train_args.seed}")

    def compute_metrics_multi(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.array([np.where(p > 0.5, 1, 0) for p in preds])
        measures, mm = train_args.measures, train_args.micro_or_macro
        result = {}
        
        metric_configs = {
            "f1": ("f1", "multilabel"),
            "precision": ("precision", "multilabel"),
            "recall": ("recall", "multilabel"),
            "accuracy": ("accuracy", "multilabel")
        }
        
        for measure in measures:
            if measure in metric_configs:
                metric_name, config_name = metric_configs[measure]
                metric = evaluate.load(metric_name, config_name=config_name)
                if measure == "accuracy":
                    result[measure] = metric.compute(predictions=preds, references=p.label_ids)[measure]
                else:
                    result[measure] = metric.compute(
                        predictions=preds, 
                        references=p.label_ids, 
                        average=mm
                    )[measure]
            else:
                raise ValueError(f"unknown measure: {measure}")
                
        return result
    
    def compute_metrics_single(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        metric = evaluate.load("accuracy", config_name="default")
        result = metric.compute(predictions=preds, references=p.label_ids)
        return result
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True
    )
    print_rank0(f"Tokenizer加载完成, vocab_size={len(tokenizer)}")
    
    train_datasets, valid_datasets = get_dataset(
        data_args,
        train_args, 
        model_args,
        tokenizer, 
    )
    
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path, 
        num_labels=model_args.num_labels,
        problem_type=train_args.problem_type,
    )
    config._attn_implementation = train_args.attn_implementation
    config.pad_token_id = tokenizer.pad_token_id
    
    
    config.id2label = {idx: label for idx, label in zip(range(model_args.num_labels), model_args.multi_classes)}
    config.label2id = {label: idx for idx, label in zip(range(model_args.num_labels), model_args.multi_classes)}
    config_dict = config.to_dict()
    config_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
    print_rank0(f"模型配置:\n{config_str}")
    
    model = get_model(config, model_args, local_rank)

    if train_args.problem_type == "multi_label_classification":
        compute_metrics = compute_metrics_multi
        def custom_preprocess(logits, labels):
            return torch.sigmoid(logits)
        preprocess_logits_for_metrics = custom_preprocess
    elif train_args.problem_type == "single_label_classification":
        compute_metrics = compute_metrics_single
        preprocess_logits_for_metrics = None
    else:
        raise ValueError(f"不支持的problem_type: {train_args.problem_type}")

    trainer = Trainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        processing_class=tokenizer,
        train_dataset=train_datasets,
        eval_dataset=valid_datasets, 
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is not None:
            print_rank0(f"发现checkpoint: {last_checkpoint}")
    
    checkpoint = None
    if train_args.resume_from_checkpoint is not None:
        checkpoint = train_args.resume_from_checkpoint
        print_rank0(f"从指定checkpoint恢复: {checkpoint}")
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
        print_rank0(f"从最新checkpoint恢复: {checkpoint}")
    
    try:
        print_rank0("=" * 50)
        print_rank0("开始训练")
        print_rank0("=" * 50)
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print_rank0(train_result)

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_datasets)
        
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        print_rank0("=" * 50)
        print_rank0("训练完成")
        print_rank0("=" * 50)
        
    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise
    
    finally:
        if dist.is_initialized():
            try:
                print_rank0("清理分布式资源...")
                dist.barrier()
                dist.destroy_process_group()
                logger.info("清理完成")
            except Exception as e:
                logger.warning(f"清理过程中出现警告: {e}")