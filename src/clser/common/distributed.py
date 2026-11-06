#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2026-06-23
# @Author : GJason
# @File : distributed.py

import os
import torch
import torch.distributed as dist
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


def is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def print_rank0(info):
    if is_rank0():
        logger.info(info)


def broadcast_from_rank0(obj):
    if not dist.is_initialized():
        return obj

    shared_obj = [obj if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(shared_obj, src=0)
    return shared_obj[0]


def cleanup_distributed():
    if not dist.is_initialized():
        return
    try:
        print_rank0("清理分布式资源...")
        dist.barrier()
        dist.destroy_process_group()
        logger.info("清理完成")
    except Exception as e:
        logger.warning(f"清理过程中出现警告: {e}")
