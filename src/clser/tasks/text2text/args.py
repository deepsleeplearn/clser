#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : args.py

from dataclasses import dataclass, field
from typeguard import typechecked
from typing import List, Optional, Literal
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from clser.tasks.classification.args import MultiClassificationDataArguments
import os


@typechecked
@dataclass
class Text2TextDataArguments(MultiClassificationDataArguments):

    system_prompt: str = field(
        default="",
        metadata={
            "help": "Default system prompt, used when an example has no 'system' key. "
                    "Empty string means no system message."
        },
    )

    max_length_threshold: int = field(
        default=1536,
        metadata={
            "help": "For causal_lm: samples whose prompt + target exceeds this many tokens are skipped."
        },
    )

    default_instruction: str = field(
        default="",
        metadata={
            "help": "Default task instruction (e.g. '请总结下面的文本。'), used when an example has no instruction key."
        },
    )

    instruction_key: str = field(
        default="instruction",
        metadata={
            "help": "jsonl key of the task instruction (optional per example)."
        },
    )

    source_key: str = field(
        default="input",
        metadata={
            "help": "jsonl key of the source text (str or list[str])."
        },
    )

    target_key: str = field(
        default="output",
        metadata={
            "help": "jsonl key of the target text."
        },
    )

    prompt_template: str = field(
        default="{instruction}\n{input}",
        metadata={
            "help": "Template used to build the user prompt. Available fields: {instruction}, {input}."
        },
    )

    source_prefix: str = field(
        default="",
        metadata={
            "help": "Prefix prepended to the encoder input for seq2seq models, e.g. 'summarize: ' for T5."
        },
    )

    use_chat_template: bool = field(
        default=True,
        metadata={
            "help": "For causal_lm: whether to format the prompt with tokenizer.apply_chat_template. "
                    "Falls back to plain concatenation when the tokenizer has no chat template."
        },
    )

    enable_thinking: Optional[bool] = field(
        default=False,
        metadata={
            "help": "For causal_lm with a Qwen3-style chat template: value passed as enable_thinking. "
                    "None means do not pass the argument."
        },
    )

    max_source_length: int = field(
        default=1024,
        metadata={
            "help": "Samples whose prompt (or encoder input) exceeds this many tokens are skipped."
        },
    )

    max_target_length: int = field(
        default=256,
        metadata={
            "help": "Samples whose target exceeds this many tokens are skipped."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        if self.valid_file_path is None:
            raise ValueError("valid_file_path must be provided: it is also used for generation eval.")

        if self.test_file_path is not None:
            if not os.path.isfile(self.test_file_path):
                raise ValueError(f"test_file_path is not a file: {self.test_file_path}")
            if not self.test_file_path.endswith(".jsonl"):
                raise ValueError(
                    f"test_file_path currently must end with .jsonl, but got "
                    f"{os.path.splitext(os.path.basename(self.test_file_path))[1]}"
                )

        if "{input}" not in self.prompt_template:
            raise ValueError(f"prompt_template must contain '{{input}}', but got {self.prompt_template!r}")

        for name in ["max_source_length", "max_target_length", "max_length_threshold"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, but got {getattr(self, name)}")


@typechecked
@dataclass
class Text2TextModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    model_type: Literal["causal_lm", "seq2seq_lm"] = field(
        default="causal_lm",
        metadata={
            "help": "causal_lm: decoder-only models such as Qwen3 / Llama (AutoModelForCausalLM). "
                    "seq2seq_lm: encoder-decoder models such as T5 / mT5 / BART (AutoModelForSeq2SeqLM)."
        },
    )

    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading tokenizer / model."},
    )


@typechecked
@dataclass
class Text2TextTrainArguments(Seq2SeqTrainingArguments):

    measures: List[str] = field(
        default_factory=lambda: ["rouge1", "rouge2", "rougeL", "bleu"],
        metadata={
            "help": "Generation metrics computed after training. "
                    "One or more of rouge1, rouge2, rougeL, bleu."
        },
    )

    metric_tokenize: Literal["char", "whitespace", "model"] = field(
        default="char",
        metadata={
            "help": "How to tokenize texts before computing ROUGE / BLEU. "
                    "char: character level (recommended for Chinese); whitespace: split on spaces (English); "
                    "model: use the model tokenizer."
        },
    )

    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = field(
        default="sdpa",
        metadata={
            "help": "The attention implementation to use."
        },
    )

    do_generation_eval: bool = field(
        default=True,
        metadata={
            "help": "Whether to reload the saved model on rank 0 after training and run generation on the "
                    "validation set (and test set when provided) to compute ROUGE / BLEU."
        },
    )

    gen_eval_max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of validation samples used for generation eval. None means all."
        },
    )

    gen_eval_batch_size: int = field(
        default=8,
        metadata={
            "help": "Batch size used during generation eval."
        },
    )

    gen_max_new_tokens: int = field(
        default=256,
        metadata={
            "help": "max_new_tokens used during generation eval."
        },
    )

    gen_num_beams: int = field(
        default=1,
        metadata={
            "help": "num_beams used during generation eval (1 means greedy)."
        },
    )

    def __post_init__(self):
        super().__post_init__()

        valid_measures = {"rouge1", "rouge2", "rougeL", "bleu"}
        if not all(measure in valid_measures for measure in self.measures):
            raise ValueError(f"measures must be one of {valid_measures}, but got {self.measures}")

        if len(list(set(self.measures))) != len(self.measures):
            raise ValueError(f"measures must be unique, but got {self.measures}")

        if self.gen_eval_batch_size <= 0:
            raise ValueError(f"gen_eval_batch_size must be positive, but got {self.gen_eval_batch_size}")

        if self.gen_max_new_tokens <= 0:
            raise ValueError(f"gen_max_new_tokens must be positive, but got {self.gen_max_new_tokens}")

        if self.gen_num_beams <= 0:
            raise ValueError(f"gen_num_beams must be positive, but got {self.gen_num_beams}")


if __name__ == "__main__":
    hf_parser = HfArgumentParser(
        [
            Text2TextDataArguments,
            Text2TextModelArguments,
            Text2TextTrainArguments,
        ]
    )

    dataargs, modelargs, trainargs = hf_parser.parse_yaml_file("args.yaml")

    print(dataargs)
    print(modelargs)
