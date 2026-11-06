#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-02-11
# @Author : GJason
# @File : run.py

import torch
import os
import json
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    EvalPrediction,
    Seq2SeqTrainer,
    default_data_collator,
    DataCollatorForSeq2Seq,
    set_seed
)
import torch.distributed as dist
from clser.common.distributed import setup_distributed, print_rank0, is_rank0, cleanup_distributed
from clser.common.args import parse_yaml_args
from clser.common.data import load_json_splits, require_columns, filter_skipped
from clser.common.trainer_utils import resolve_checkpoint, save_train_result, final_evaluate, banner
from clser.tasks.text2text.args import (
    Text2TextDataArguments,
    Text2TextModelArguments,
    Text2TextTrainArguments,
)
from clser.tasks.text2text.preprocess import (
    IGNORE_INDEX,
    build_preprocess_functions,
    build_eval_prompt_text,
    build_target,
    causal_add_special_tokens,
    clean_generation,
)
from clser.tasks.text2text.metrics import compute_text_metrics
from clser.utils.logger import get_logger

logger = get_logger()

MODEL_COLUMNS = {"input_ids", "attention_mask", "labels"}


def load_tokenizer(model_args: "Text2TextModelArguments"):
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither pad_token nor eos_token; cannot pad batches.")
        tokenizer.pad_token = tokenizer.eos_token
        print_rank0(f"tokenizer 没有 pad_token，已设置为 eos_token: {tokenizer.eos_token}")
    return tokenizer


def get_dataset(data_args: "Text2TextDataArguments",
                train_args: "Text2TextTrainArguments",
                model_args: "Text2TextModelArguments",
                tokenizer: "AutoTokenizer"
                ):

    with train_args.main_process_first(desc="pre-process dataset", local=True):

        format_dataset = load_json_splits(data_args.train_file_path, data_args.valid_file_path)
        require_columns(format_dataset, [data_args.source_key, data_args.target_key])

        preprocess_func, preprocess_func_batched = build_preprocess_functions(
            data_args=data_args,
            model_type=model_args.model_type,
            tokenizer=tokenizer,
            logger=logger,
        )
        base_map_kwargs = {
            "function": preprocess_func_batched if data_args.use_batched_preprocess else preprocess_func,
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

        train_datasets = format_dataset["train"].map(**base_map_kwargs, desc="Running tokenizer on train dataset")
        if data_args.shuffle:
            train_datasets = train_datasets.shuffle(seed=data_args.shuffle_seed)
        train_datasets = filter_skipped(train_datasets, "训练集", data_args.num_processing, MODEL_COLUMNS)

        valid_datasets = format_dataset["valid"].map(**base_map_kwargs, desc="Running tokenizer on valid dataset")
        valid_datasets = filter_skipped(valid_datasets, "验证集", data_args.num_processing, MODEL_COLUMNS)

        if len(train_datasets) == 0:
            raise ValueError("训练集过滤后为空，请检查数据格式或放宽长度阈值。")

        lengths = [len(x) for x in train_datasets["input_ids"]]
        print_rank0(
            f"训练集样本数: {len(train_datasets)}, 序列长度 min/mean/max: "
            f"{int(np.min(lengths))}/{float(np.mean(lengths)):.1f}/{int(np.max(lengths))}"
        )

        return train_datasets, valid_datasets


def get_model_class(model_type: str):
    if model_type == "causal_lm":
        return AutoModelForCausalLM
    if model_type == "seq2seq_lm":
        return AutoModelForSeq2SeqLM
    raise ValueError(f"不支持的 model_type: {model_type}")


def get_model(config: "AutoConfig",
              model_args: "Text2TextModelArguments",
              local_rank: int
              ):

    try:
        model_cls = get_model_class(model_args.model_type)
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
            print_rank0(f"模型所在设备: {model.device}")

        model.train()
        return model

    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise


def read_jsonl(path: str):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def run_generation_eval(data_args: "Text2TextDataArguments",
                        model_args: "Text2TextModelArguments",
                        train_args: "Text2TextTrainArguments",
                        tokenizer: "AutoTokenizer",
                        local_rank: int):
    if not train_args.do_generation_eval:
        print_rank0("跳过生成评估: do_generation_eval=False")
        return None

    if dist.is_initialized():
        dist.barrier()

    results = None
    if is_rank0():
        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        print_rank0(f"从 {train_args.output_dir} 重新加载模型用于生成评估 (device={device})")
        model_cls = get_model_class(model_args.model_type)
        model = model_cls.from_pretrained(
            train_args.output_dir,
            dtype=torch.bfloat16,
            trust_remote_code=model_args.trust_remote_code,
        ).to(device)
        model.eval()

        original_padding_side = tokenizer.padding_side
        if model_args.model_type == "causal_lm":
            tokenizer.padding_side = "left"

        results = {}
        splits = [("valid", data_args.valid_file_path)]
        if data_args.test_file_path is not None:
            splits.append(("test", data_args.test_file_path))

        try:
            for split_name, path in splits:
                examples = read_jsonl(path)
                if train_args.gen_eval_max_samples is not None:
                    examples = examples[: train_args.gen_eval_max_samples]

                prompts, references, kept = [], [], []
                for example in examples:
                    try:
                        prompt = build_eval_prompt_text(example, data_args, model_args.model_type, tokenizer)
                    except Exception as e:
                        logger.warning(f"[{split_name}] 构建 prompt 失败, 跳过: {e}")
                        continue
                    try:
                        reference = build_target(example, data_args)
                    except Exception:
                        reference = None
                    prompts.append(prompt)
                    references.append(reference)
                    kept.append(example)

                predictions = []
                batch_size = train_args.gen_eval_batch_size
                for start in range(0, len(prompts), batch_size):
                    batch_prompts = prompts[start:start + batch_size]
                    encoded = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=(
                            True if model_args.model_type == "seq2seq_lm"
                            else causal_add_special_tokens(data_args, tokenizer)
                        ),
                    ).to(device)
                    with torch.no_grad():
                        generated = model.generate(
                            **encoded,
                            max_new_tokens=train_args.gen_max_new_tokens,
                            num_beams=train_args.gen_num_beams,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    if model_args.model_type == "causal_lm":
                        generated = generated[:, encoded["input_ids"].shape[1]:]
                    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                    predictions.extend(clean_generation(text) for text in decoded)
                    print_rank0(f"[{split_name}] 生成进度: {min(start + batch_size, len(prompts))}/{len(prompts)}")

                pred_path = os.path.join(train_args.output_dir, f"generation_{split_name}_predictions.jsonl")
                with open(pred_path, "w", encoding="utf-8") as f:
                    for prompt, prediction, reference in zip(prompts, predictions, references):
                        f.write(json.dumps(
                            {"prompt": prompt, "prediction": prediction, "reference": reference},
                            ensure_ascii=False,
                        ) + "\n")
                print_rank0(f"[{split_name}] 生成结果已保存到 {pred_path}")

                if references and all(r is not None for r in references):
                    metrics = compute_text_metrics(
                        predictions=predictions,
                        references=references,
                        measures=train_args.measures,
                        tokenize_mode=train_args.metric_tokenize,
                        tokenizer=tokenizer,
                    )
                    metrics = {f"gen_{split_name}_{k}": v for k, v in metrics.items()}
                    metrics[f"gen_{split_name}_samples"] = len(predictions)
                    results[split_name] = metrics
                    print_rank0(f"[{split_name}] 生成评估结果: {metrics}")
                else:
                    print_rank0(f"[{split_name}] 缺少参考答案，仅保存生成结果，不计算指标")

            metrics_path = os.path.join(train_args.output_dir, "generation_eval_results.json")
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print_rank0(f"生成评估指标已保存到 {metrics_path}")
        finally:
            tokenizer.padding_side = original_padding_side
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.barrier()
    return results


def run_text2text(yaml_path: str):

    local_rank, rank, world_size = setup_distributed()
    print_rank0(f"Starting text2text training on {world_size} GPU(s)")

    data_args, model_args, train_args = parse_yaml_args(
        yaml_path, (Text2TextDataArguments, Text2TextModelArguments, Text2TextTrainArguments)
    )

    if train_args.seed is not None:
        set_seed(train_args.seed)
        print_rank0(f"Random seed set to {train_args.seed}")

    if model_args.model_type == "causal_lm" and train_args.predict_with_generate:
        logger.warning(
            "causal_lm 的训练样本已包含答案，Trainer 内的 predict_with_generate 不可用，已强制关闭；"
            "生成指标由训练结束后的 run_generation_eval 计算。"
        )
        train_args.predict_with_generate = False

    tokenizer = load_tokenizer(model_args)
    print_rank0(f"Tokenizer加载完成, vocab_size={len(tokenizer)}, chat_template={'有' if tokenizer.chat_template else '无'}")

    train_datasets, valid_datasets = get_dataset(
        data_args,
        train_args,
        model_args,
        tokenizer,
    )

    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    config._attn_implementation = train_args.attn_implementation
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id
    if hasattr(config, "use_cache"):
        config.use_cache = False
        print_rank0("已关闭 config.use_cache，用于降低训练显存占用。")
    if train_args.gradient_checkpointing:
        print_rank0(
            f"已启用 gradient checkpointing, kwargs={train_args.gradient_checkpointing_kwargs}"
        )

    print_rank0(f"模型配置:\n{json.dumps(config.to_dict(), indent=2, ensure_ascii=False, default=str)}")

    model = get_model(config, model_args, local_rank)

    if data_args.use_dcwp:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=IGNORE_INDEX,
            padding=True,
            pad_to_multiple_of=data_args.pad_to_multiple_of,
            return_tensors="pt",
        )
    else:
        data_collator = default_data_collator

    def compute_metrics_generation(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.where(preds != IGNORE_INDEX, preds, tokenizer.pad_token_id)
        labels = np.where(p.label_ids != IGNORE_INDEX, p.label_ids, tokenizer.pad_token_id)
        decoded_preds = [clean_generation(t) for t in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [t.strip() for t in tokenizer.batch_decode(labels, skip_special_tokens=True)]
        return compute_text_metrics(
            predictions=decoded_preds,
            references=decoded_labels,
            measures=train_args.measures,
            tokenize_mode=train_args.metric_tokenize,
            tokenizer=tokenizer,
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        compute_metrics=compute_metrics_generation if train_args.predict_with_generate else None,
        data_collator=data_collator,
        processing_class=tokenizer,
        train_dataset=train_datasets,
        eval_dataset=valid_datasets,
    )

    checkpoint = resolve_checkpoint(train_args)

    try:
        banner("开始训练")

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        print_rank0(train_result)

        save_train_result(trainer, train_result, len(train_datasets))
        final_evaluate(trainer, valid_datasets)

        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        run_generation_eval(data_args, model_args, train_args, tokenizer, local_rank)

        banner("训练完成")

    except Exception as e:
        logger.error(f"训练过程中出错: {e}")
        raise

    finally:
        cleanup_distributed()
