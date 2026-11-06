#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : args.py

from dataclasses import dataclass, field
from typeguard import typechecked
from typing import List, Optional, Literal
from transformers import HfArgumentParser, TrainingArguments
from clser.tasks.classification.args import MultiClassificationDataArguments


@typechecked
@dataclass
class RerankDataArguments(MultiClassificationDataArguments):

    system_prompt: str = field(
        default="",
        metadata={
            "help": "System prompt for llm rerankers. Empty string uses the Qwen3-Reranker default."
        },
    )

    max_length_threshold: int = field(
        default=512,
        metadata={
            "help": "Maximum total length of query + passage (+ prompt) in tokens; longer inputs are truncated."
        },
    )

    query_key: str = field(
        default="query",
        metadata={"help": "jsonl key of the query."},
    )

    pos_key: str = field(
        default="pos",
        metadata={"help": "jsonl key of positive passages (str or list[str])."},
    )

    neg_key: str = field(
        default="neg",
        metadata={"help": "jsonl key of negative passages (str or list[str])."},
    )

    instruction_key: str = field(
        default="prompt",
        metadata={"help": "jsonl key of the optional task instruction."},
    )

    default_instruction: str = field(
        default="Given a web search query, retrieve relevant passages that answer the query.",
        metadata={"help": "Instruction used when an example has no instruction key."},
    )

    train_group_size: int = field(
        default=8,
        metadata={"help": "Number of passages per query during training: 1 positive + (group_size - 1) negatives."},
    )

    eval_group_size: Optional[int] = field(
        default=None,
        metadata={"help": "Group size for the validation set. None means train_group_size."},
    )

    neg_sample_seed: int = field(
        default=42,
        metadata={"help": "Seed used to sample positives / negatives for each query."},
    )

    max_query_length: int = field(
        default=64,
        metadata={"help": "Queries longer than this many tokens are truncated."},
    )

    max_passage_length: int = field(
        default=448,
        metadata={"help": "Passages longer than this many tokens are truncated."},
    )

    enable_thinking: Optional[bool] = field(
        default=False,
        metadata={"help": "For llm rerankers with a Qwen3-style chat template: value passed as enable_thinking."},
    )

    def __post_init__(self):
        super().__post_init__()

        if self.valid_file_path is None:
            raise ValueError("valid_file_path must be provided.")

        if self.train_group_size < 2:
            raise ValueError(f"train_group_size must be >= 2, but got {self.train_group_size}")

        if self.eval_group_size is None:
            self.eval_group_size = self.train_group_size
        elif self.eval_group_size < 2:
            raise ValueError(f"eval_group_size must be >= 2, but got {self.eval_group_size}")

        for name in ["max_query_length", "max_passage_length", "max_length_threshold"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, but got {getattr(self, name)}")


@typechecked
@dataclass
class RerankModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_type: Literal["cross_encoder", "llm"] = field(
        default="cross_encoder",
        metadata={
            "help": "cross_encoder: BERT-style models scored by a 1-dim classification head "
                    "(bge-reranker etc.). llm: decoder-only models scored by logit(yes) - logit(no) "
                    "(Qwen3-Reranker style)."
        },
    )

    yes_token: str = field(
        default="yes",
        metadata={"help": "llm only: token whose logit represents 'relevant'."},
    )

    no_token: str = field(
        default="no",
        metadata={"help": "llm only: token whose logit represents 'not relevant'."},
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading tokenizer / model."},
    )


@typechecked
@dataclass
class RerankTrainArguments(TrainingArguments):

    loss_type: Literal["listwise", "pointwise"] = field(
        default="listwise",
        metadata={
            "help": "listwise: softmax cross entropy over each group (positive at index 0). "
                    "pointwise: binary cross entropy on every (query, passage) pair."
        },
    )

    temperature: float = field(
        default=1.0,
        metadata={"help": "Scores are divided by this value before the listwise softmax."},
    )

    measures: List[str] = field(
        default_factory=lambda: ["accuracy", "mrr", "ndcg"],
        metadata={"help": "Ranking metrics over each eval group. One or more of accuracy, mrr, ndcg."},
    )

    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = field(
        default="sdpa",
        metadata={"help": "The attention implementation to use."},
    )

    def __post_init__(self):
        super().__post_init__()

        valid_measures = {"accuracy", "mrr", "ndcg"}
        if not all(measure in valid_measures for measure in self.measures):
            raise ValueError(f"measures must be one of {valid_measures}, but got {self.measures}")

        if len(list(set(self.measures))) != len(self.measures):
            raise ValueError(f"measures must be unique, but got {self.measures}")

        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, but got {self.temperature}")


if __name__ == "__main__":
    hf_parser = HfArgumentParser(
        [
            RerankDataArguments,
            RerankModelArguments,
            RerankTrainArguments,
        ]
    )

    dataargs, modelargs, trainargs = hf_parser.parse_yaml_file("args.yaml")

    print(dataargs)
    print(modelargs)
