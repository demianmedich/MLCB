# coding=utf-8
import warnings

import torch.distributed as dist
from loguru import logger


def rank_zero_warn(msg: str) -> None:
    """print on"""
    if dist.is_initialized():
        if dist.get_rank(dist.group.WORLD) == 0:
            logger.warning(msg)
    else:
        logger.warning(msg)
