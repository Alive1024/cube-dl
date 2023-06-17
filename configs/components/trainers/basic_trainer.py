from typing import Union, Iterable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import Logger

from config_sys import trainer_getter, shared_config


@trainer_getter
def get_trainer_instance(logger: Union[Logger, Iterable[Logger], bool]):
    return pl.Trainer(
        # accelerator="mps",
        # devices=1,
        max_epochs=shared_config.get("fit_max_epochs"),
        callbacks=[
            RichProgressBar(leave=True)
        ],
        logger=logger,
    )
