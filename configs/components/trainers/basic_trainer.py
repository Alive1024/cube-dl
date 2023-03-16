from typing import Union, Iterable

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import Logger


def get_trainer_instance(logger: Union[Logger, Iterable[Logger], bool]):
    return pl.Trainer(
        # accelerator="mps",
        # devices=1,
        max_epochs=5,
        callbacks=[
            RichProgressBar(leave=True)
        ],
        logger=logger,
    )
