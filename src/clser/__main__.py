import sys
import os
from clser.utils.logger import get_logger
from clser.core.run_classification import run_classification
import torch.distributed as dist
logger = get_logger()


def main():
    try:
        run_classification()
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    main()