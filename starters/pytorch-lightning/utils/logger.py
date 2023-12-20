from collections.abc import Iterable
from functools import partial

from cube.c3lyr import Run
from pytorch_lightning.loggers import (
    CSVLogger,
    TensorBoardLogger,
    WandbLogger,  # noqa
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only


def get_csv_logger(run: Run) -> CSVLogger:
    belonging_exp = run.belonging_exp
    belonging_proj = belonging_exp.belonging_proj
    # Logs are saved to `os.path.join(save_dir, name, version)`.
    return CSVLogger(
        save_dir=belonging_proj.proj_dir,
        name=belonging_exp.dirname,
        version=run.dirname,
    )


def get_tensorboard_logger(run: Run) -> TensorBoardLogger:
    belonging_exp = run.belonging_exp
    belonging_proj = belonging_exp.belonging_proj
    return TensorBoardLogger(
        save_dir=belonging_proj.proj_dir,
        name=belonging_exp.dirname,
        version=run.dirname,
    )


def get_wandb_logger(run: Run) -> WandbLogger:
    belonging_exp = run.belonging_exp
    belonging_proj = belonging_exp.belonging_proj
    wandb_logger_partial = partial(
        WandbLogger,
        save_dir=belonging_proj.proj_dir,
        name=run.dirname,  # display name for the run
        # The name of the project to which this run will belong:
        project=belonging_proj.dirname,
        group=belonging_exp.dirname,  # use exp_name to group runs
        job_type=run.job_type,
        id=run.global_id,
    )
    return wandb_logger_partial(resume="must") if run.is_resuming else wandb_logger_partial()


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Some loggers support for logging hyper-parameters.


def add_hparams_to_wandb_logger(loggers: Iterable | bool, hparams: dict):
    """Log hyper-parameters using `WandbLogger`."""
    # Execute only on rank 0, more details at: https://github.com/Lightning-AI/lightning/issues/13166
    if rank_zero_only.rank == 0 and loggers:  # noqa
        for logger in loggers:
            if isinstance(logger, WandbLogger):
                # Note: use directly wandb module here (i.e. `wandb.config.update(hparams)`)
                # will trigger an error: "wandb.errors.Error: You must call wandb.init() before wandb.config.update"
                logger.experiment.config.update(hparams)
