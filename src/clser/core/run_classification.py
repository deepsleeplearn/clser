#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-04-02
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
    DataCollatorWithPadding,
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
from clser.core.preprocess import build_preprocess_functions
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


def get_dataset(data_args: "MultiClassificationDataArguments", 
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

        preprocess_func, preprocess_func_batched = build_preprocess_functions(
            data_args=data_args,
            train_args=train_args,
            tokenizer=tokenizer,
            mlb=mlb,
            logger=logger,
        )
        selected_preprocess_func = (
            preprocess_func_batched if data_args.use_batched_preprocess else preprocess_func
        )
        base_map_kwargs = {
            "function": selected_preprocess_func,
            "num_proc": data_args.num_processing,
        }
        if data_args.use_batched_preprocess:
            base_map_kwargs["batched"] = True
            base_map_kwargs["batch_size"] = data_args.preprocess_batch_size
            print_rank0(
                f"启用批量预处理: batch_size={data_args.preprocess_batch_size}, num_proc={data_args.num_processing}"
            )
        else:
            print_rank0(f"启用单样本预处理: num_proc={data_args.num_processing}")

        train_datasets = format_dataset["train"]
        train_datasets = train_datasets.map(
            **base_map_kwargs,
            desc="Running tokenizer on train dataset",
        )

        if data_args.train_remove_columns is not None:
            train_datasets = train_datasets.remove_columns(data_args.train_remove_columns)

        if data_args.shuffle:
            train_datasets = train_datasets.shuffle(seed=data_args.shuffle_seed)

        valid_datasets = format_dataset["valid"]
        valid_map_kwargs = dict(base_map_kwargs)
        valid_map_kwargs["desc"] = "Running tokenizer on valid dataset"
        if data_args.valid_remove_columns is not None:
            valid_map_kwargs["remove_columns"] = data_args.valid_remove_columns
        valid_datasets = valid_datasets.map(**valid_map_kwargs)

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
    if hasattr(config, "use_cache"):
        config.use_cache = False
        print_rank0("已关闭 config.use_cache，用于降低分类训练显存占用。")
    if train_args.gradient_checkpointing:
        print_rank0(
            f"已启用 gradient checkpointing, kwargs={train_args.gradient_checkpointing_kwargs}"
        )

    config_dict = config.to_dict()
    config_str = json.dumps(config_dict, indent=2, ensure_ascii=False)
    print_rank0(f"模型配置:\n{config_str}")


    if train_args.problem_type == "multi_label_classification":
        compute_metrics = compute_metrics_multi
        def custom_preprocess(logits, labels):
            return torch.sigmoid(logits).detach()
        preprocess_logits_for_metrics = custom_preprocess
    elif train_args.problem_type == "single_label_classification":
        compute_metrics = compute_metrics_single
        preprocess_logits_for_metrics = None
    else:
        raise ValueError(f"不支持的problem_type: {train_args.problem_type}")
    
    if data_args.use_dcwp:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=data_args.pad_to_multiple_of,
            return_tensors="pt"
        )
    else:
        data_collator = default_data_collator

    def log_and_save_train_metrics(trainer: Trainer, train_result):
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_datasets)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        return metrics

    def run_final_evaluation(trainer: Trainer):
        if not train_args.do_eval:
            print_rank0("跳过最终评估: do_eval=False")
            return None

        final_eval_metrics = trainer.evaluate(eval_dataset=valid_datasets, metric_key_prefix="final_eval")
        final_eval_metrics["final_eval_samples"] = len(valid_datasets)

        trainer.log_metrics("final_eval", final_eval_metrics)
        trainer.save_metrics("final_eval", final_eval_metrics)

        print_rank0(f"最终评估结果: {final_eval_metrics}")
        print_rank0("=" * 50)
        print_rank0("最终评估完成")
        print_rank0("=" * 50)
        return final_eval_metrics

    def broadcast_from_rank0(obj):
        if not dist.is_initialized():
            return obj

        shared_obj = [obj if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(shared_obj, src=0)
        return shared_obj[0]


    if not train_args.use_hyperparameter_search:
        model = get_model(config, model_args, local_rank)

        trainer = Trainer(
            model=model,
            args=train_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
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

            trainer.save_model()
            log_and_save_train_metrics(trainer, train_result)
            trainer.save_state()
            run_final_evaluation(trainer)
            
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
    else:

        def model_init(trial=None):
            model = Qwen3ForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                dtype=torch.bfloat16,
            )
            return model

        trainer = Trainer(
            model_init=model_init,
            args=train_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            processing_class=tokenizer,
            train_dataset=train_datasets,
            eval_dataset=valid_datasets, 
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        print_rank0(f"开始超参搜索: n_trials={train_args.hp_n_trials}, backend={train_args.hp_search_backend}")

        if not train_args.do_eval:
            raise ValueError("use_hyperparameter_search=True 时必须设置 do_eval=True。")
        
        def hp_space(trial):
            return_dict = {}
            if "lr" in train_args.hp_dict:
                return_dict["learning_rate"] = trial.suggest_float(
                    "learning_rate", float(train_args.hp_dict["lr"]["hp_lr_min"]), float(train_args.hp_dict["lr"]["hp_lr_max"]), log=True
                    )
            if "per_device_train_batch_size" in train_args.hp_dict:
                assert isinstance(train_args.hp_dict["per_device_train_batch_size"], list), "Batch size just support specified num."
                return_dict["per_device_train_batch_size"] = trial.suggest_categorical(
                        "per_device_train_batch_size", train_args.hp_dict["per_device_train_batch_size"]
                    )
            if "warmup_ratio" in train_args.hp_dict:
                return_dict["warmup_ratio"] = trial.suggest_float(
                        "warmup_ratio", float(train_args.hp_dict["warmup_ratio"]["hp_warmup_ratio_min"]), float(train_args.hp_dict["warmup_ratio"]["hp_warmup_ratio_max"])
                    )
            if "weight_decay" in train_args.hp_dict:
                return_dict["weight_decay"] = trial.suggest_float(
                        "weight_decay", float(train_args.hp_dict["weight_decay"]["hp_weight_decay_min"]), float(train_args.hp_dict["weight_decay"]["hp_weight_decay_max"])
                    )
            return return_dict

        def compute_objective(metrics):
            if train_args.hp_metric_for_best not in metrics:
                available_metrics = ", ".join(sorted(metrics.keys()))
                raise ValueError(
                    f"hp_metric_for_best={train_args.hp_metric_for_best} 不在评估结果中，可用指标: {available_metrics}"
                )
            return metrics[train_args.hp_metric_for_best]
        
        try:

            best_run = trainer.hyperparameter_search(
                direction=train_args.hp_direction,
                backend=train_args.hp_search_backend,
                hp_space=hp_space,
                n_trials=train_args.hp_n_trials,
                compute_objective=compute_objective,
            )

            best_run_payload = None
            if best_run is not None:
                best_run_payload = {
                    "run_id": best_run.run_id,
                    "objective": best_run.objective,
                    "hyperparameters": best_run.hyperparameters,
                }
                print_rank0(f"最优 trial: {best_run.run_id}")
                print_rank0(f"最优超参: {best_run.hyperparameters}")
                print_rank0(f"最优 {train_args.hp_metric_for_best}: {best_run.objective}")

            best_run_payload = broadcast_from_rank0(best_run_payload)
            if best_run_payload is None:
                raise RuntimeError("超参搜索未返回最优结果。")

            print_rank0("用最优超参重新训练完整模型...")
            print_rank0(f"最终重训使用超参: {best_run_payload['hyperparameters']}")
            for key, value in best_run_payload["hyperparameters"].items():
                setattr(trainer.args, key, value)

            train_result = trainer.train()
            trainer.save_model()
            log_and_save_train_metrics(trainer, train_result)
            trainer.save_state()
            run_final_evaluation(trainer)

            hp_save_path = os.path.join(train_args.output_dir, "best_hyperparameters.json")
            with open(hp_save_path, "w") as f:
                json.dump(best_run_payload, f, indent=2, ensure_ascii=False)
            print_rank0(f"最优超参已保存到 {hp_save_path}")
        except Exception as e:
            logger.error(f"训练过程中出错: {e}")
            raise
        finally:
            if dist.is_initialized():
                try:
                    dist.barrier()
                    dist.destroy_process_group()
                except Exception as e:
                    logger.warning(f"清理过程中出现警告: {e}")
