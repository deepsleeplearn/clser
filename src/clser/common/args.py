#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : args.py

import os
import yaml
from transformers import HfArgumentParser


def load_yaml(args_file: str) -> dict:
    assert os.path.exists(args_file) and os.path.isfile(args_file), \
        f"yaml_path does not exist: {args_file}"
    with open(args_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_yaml_args(args_file: str, dataclass_types):
    yaml_dict = load_yaml(args_file)
    yaml_dict.pop("task", None)
    hf_parser = HfArgumentParser(dataclass_types)
    return hf_parser.parse_dict(yaml_dict)
