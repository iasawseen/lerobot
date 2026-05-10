# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import logging
from pathlib import Path

from termcolor import colored

from lerobot.configs.train import TrainPipelineConfig


class TensorBoardLogger:
    """Thin wrapper around ``torch.utils.tensorboard.SummaryWriter`` that
    mirrors :class:`lerobot.common.wandb_utils.WandBLogger.log_dict` so the
    training loop can drive both loggers from the same call sites.
    """

    def __init__(self, cfg: TrainPipelineConfig):
        from torch.utils.tensorboard import SummaryWriter

        log_dir = Path(cfg.output_dir) / cfg.tensorboard.subdir
        log_dir.mkdir(parents=True, exist_ok=True)
        # flush_secs=10 (vs SummaryWriter default 120) so scalars become
        # visible to a live `tensorboard --logdir` within ~10s of being logged.
        self._writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)
        self.log_dir = log_dir
        logging.info(
            colored("TensorBoard logs:", "blue", attrs=["bold"])
            + f" {log_dir}  (run `tensorboard --logdir {log_dir.parent}`)"
        )

    def log_dict(self, d: dict, step: int, mode: str = "train") -> None:
        if mode not in {"train", "eval"}:
            raise ValueError(mode)
        for k, v in d.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            self._writer.add_scalar(f"{mode}/{k}", float(v), step)

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()
