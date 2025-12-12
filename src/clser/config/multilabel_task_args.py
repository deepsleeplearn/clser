#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025-12-12
# @Author : guojian
# @File : multilabel_task_args.py

from dataclasses import dataclass, field
from typeguard import typechecked
from typing import List, Optional, Literal
from transformers import HfArgumentParser, TrainingArguments
import os

@typechecked
@dataclass
class MultiClassificationDataArguments:
    
    train_file_path: str = field(
        metadata={
            "help": "The training data file (.jsonl)."
        },
    )

    valid_file_path: Optional[str] = field(
        metadata={
            "help": "The validation data file (.jsonl)."
        },
    )

    test_file_path: Optional[str] = field(
        metadata={
            "help": "The test data file (.jsonl)."
        },
    )
    
    task_name: str = field(
        default="state_model", 
        metadata={
            "help": "The name of the task."
        },
    )

    shuffle: bool = field(
        default=True,
        metadata={
            "help": "Whether to shuffle the training data."
        },
    )

    shuffle_seed: int = field(
        default=42,
        metadata={
            "help": "Random seed that will be used to shuffle the training data."
        },
    )
    
    label_key: str = field(
        default="label",
        metadata={
            "help": "jsonl data label key."
        },
    )

    train_remove_columns: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "The columns to remove from the training data."
        },
    )

    valid_remove_columns: Optional[List[str]] = field(
        default_factory=lambda: None,
        metadata={
            "help": "The columns to remove from the valid data."
        },
    )

    num_processing: int = field(
        default=16,
        metadata={
            "help": "The number of processing."
        },
    )

    max_length_threshold: int = field(
        default=1024,
        metadata={
            "help": "The maximum length of a sequence can be used. "
        },
    )

    def __post_init__(self):

        if not os.path.exists(self.train_file_path):
            raise ValueError(f"train_file_path does not exist: {self.train_file_path}")
        
        if not os.path.isfile(self.train_file_path):
            raise ValueError(f"train_file_path is not a file: {self.train_file_path}")
        
        if not self.train_file_path.endswith(".jsonl"):
            raise ValueError(f"train_file_path currently must end with .jsonl, but got {os.path.splitext(os.path.basename(self.train_file_path))[1]}")
        
        if self.valid_file_path is not None:
            if not os.path.exists(self.valid_file_path):
                raise ValueError(f"valid_file_path does not exist: {self.valid_file_path}")
            if not os.path.isfile(self.valid_file_path):
                raise ValueError(f"valid_file_path is not a file: {self.valid_file_path}")
            if not self.valid_file_path.endswith(".jsonl"):
                raise ValueError(f"valid_file_path currently must end with .jsonl, but got {os.path.splitext(os.path.basename(self.valid_file_path))[1]}")
        
        

@typechecked
@dataclass
class MultiClassificationModelArguments:

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
    
    multi_classes: List[str] = field(
        metadata={
            "help": "Whether to use multi-classes classification."
            },
        )
    
    def __post_init__(self):

        self.num_labels = len(self.multi_classes)
    






@typechecked
@dataclass
class MultiClassificationTrainArguments(TrainingArguments):
    
    problem_type: Literal["multi_label_classification", "single_label_classification"] = field(
        default="multi_label_classification",
        metadata={"help": "The type of problem, one of classification, regression, or multi_label_classification"},
    )

    measures: List[str] = field(
        default_factory=lambda: ["f1"],
        metadata={
            "help": "The measures to measure."
            },
        )
    
    micro_or_macro: Literal["micro", "macro"] = field(
        default="micro",
        metadata={
            "help": "The micro or macro to measure."
            },
        )

    attn_implementation: Literal["eager", "flash_attention_2", "sdpa"] = field(
        default="flash_attention_2",
        metadata={
            "help": "The attention implementation to use."
            },
        )
    
    def __post_init__(self):
        super().__post_init__()

        valid_measures = {"accuracy", "precision", "recall", "f1"}
        if not all(measure in valid_measures for measure in self.measures):
            raise ValueError(f"measures must be one of {valid_measures}, but got {self.measures}")
        
        if len(list(set(self.measures))) != len(self.measures):
            raise ValueError(f"measures must be unique, but got {self.measures}")



            
@typechecked
@dataclass
class MultiClassificationInferArguments:
    
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
        )
    
    multi_classes: List[str] = field(
        metadata={
            "help": "Whether to use multi-classes classification."
            },
        )
    
    problem_type: str = field(
        default="multi_label_classification",
        metadata={"help": "The type of problem, one of classification, regression, or multi_label_classification"},
    )
    
    max_length_threshold: int = field(
        default=1024,
        metadata={
            "help": "The maximum length of a sequence can be used. "
        },
    )
    def __post_init__(self):

        self.num_labels = len(self.multi_classes)
    
    
            

if __name__ == "__main__":
    hf_parser = HfArgumentParser(
        [
            MultiClassificationDataArguments, 
            MultiClassificationModelArguments,
            MultiClassificationTrainArguments
            ]
            )

    dataargs, modelargs, trainargs = hf_parser.parse_yaml_file("args.yaml")

    print(dataargs)
    print(modelargs)