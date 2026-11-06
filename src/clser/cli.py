#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-11-06
# @Author : GJason
# @File : cli.py

import sys
import importlib
from clser.common.args import load_yaml
from clser.common.distributed import cleanup_distributed
from clser.utils.logger import get_logger

logger = get_logger()

TASKS = {
    "classification": ("clser.tasks.classification.run", "run_classification"),
    "text2text": ("clser.tasks.text2text.run", "run_text2text"),
    "rerank": ("clser.tasks.rerank.run", "run_rerank"),
}

USAGE = (
    "用法:\n"
    "  python -m clser <task> <config.yaml>\n"
    "  python -m clser <config.yaml>          (yaml 中需包含 task 字段)\n"
    f"可选 task: {', '.join(TASKS)}"
)


def resolve(argv):
    if len(argv) == 1:
        config = argv[0]
        task = load_yaml(config).get("task")
        if task is None:
            raise SystemExit(f"{config} 中没有 task 字段。\n{USAGE}")
    elif len(argv) == 2:
        task, config = argv
    else:
        raise SystemExit(USAGE)

    if task not in TASKS:
        raise SystemExit(f"未知 task: {task!r}\n{USAGE}")
    return task, config


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    task, config = resolve(argv)

    module_name, func_name = TASKS[task]
    run = getattr(importlib.import_module(module_name), func_name)

    try:
        run(config)
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        cleanup_distributed()
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        cleanup_distributed()
        raise


if __name__ == "__main__":
    main()
