import os
import os.path as osp
from typing import Optional, Callable, Union, Iterable, List
import inspect
import shutil
import re
from collections import OrderedDict
import json

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import Logger, CSVLogger, TensorBoardLogger, WandbLogger

from entities import Run


class Config:
    def __init__(self, *,
                 task_wrapper_getter: Callable[[], pl.LightningModule],
                 data_wrapper_getter: Callable[[], pl.LightningDataModule],

                 default_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 tuner: Tuner = None,
                 fit_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 validate_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 test_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 predict_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 ):
        if (not default_trainer_getter) and (not fit_trainer_getter) and (not validate_trainer_getter) \
                and (not test_trainer_getter) and (not predict_trainer_getter):
            raise RuntimeError("The trainer getters can't be all None.")

        self.task_wrapper_getter = task_wrapper_getter
        self.data_wrapper_getter = data_wrapper_getter
        self.default_trainer_getter = default_trainer_getter
        self.tuner = tuner
        self.fit_trainer_getter = fit_trainer_getter
        self.validate_trainer_getter = validate_trainer_getter
        self.test_trainer_getter = test_trainer_getter
        self.predict_trainer_getter = predict_trainer_getter

        self.task_wrapper: Optional[pl.LightningModule] = None
        self.data_wrapper: Optional[pl.LightningDataModule] = None
        self.fit_trainer: Optional[pl.Trainer] = None
        self.validate_trainer: Optional[pl.Trainer] = None
        self.test_trainer: Optional[pl.Trainer] = None
        self.predict_trainer: Optional[pl.Trainer] = None

    @staticmethod
    def _setup_logger(logger_arg: str, run: Run) \
            -> Union[Logger, Iterable[Logger], bool]:
        loggers: List[str] = re.split(r"[\s,]+", logger_arg)
        logger_instances = []
        for logger_str in loggers:
            logger_name = logger_str.lower()
            if logger_name == "true":
                return CSVLogger(save_dir=run.proj_dir, name=run.exp_name, version=run.name)
            elif logger_name == "false":
                return False
            else:
                if logger_name == "csv":
                    logger_instances.append(CSVLogger(save_dir=run.proj_dir,
                                                      name=run.exp_name,
                                                      version=run.name))
                elif logger_name == "tensorboard":
                    logger_instances.append(TensorBoardLogger(save_dir=run.proj_dir,
                                                              name=run.exp_name,
                                                              version=run.name))
                elif logger_name == "wandb":
                    import wandb
                    # UserWarning: There is a wandb run already in progress and newly created instances of
                    # `WandbLogger` will reuse this run. If this is not desired, call `wandb.finish()` before
                    # instantiating `WandbLogger`.
                    wandb.finish()
                    if run.is_resuming:
                        wandb_logger = WandbLogger(save_dir=run.proj_dir,
                                                   name=run.name,  # display name for the run
                                                   # version=job_type,
                                                   # The name of the project to which this run will belong:
                                                   project=osp.split(run.proj_dir)[1],
                                                   group=run.exp_name,  # use exp_name to group runs
                                                   job_type=run.name.split('_')[-1],
                                                   id=run.get_extra_record_data("wandb_run_id"), resume="must"
                                                   )
                    else:
                        # Generate a run id and save it to the record file for future fit resuming.
                        wandb_run_id = wandb.util.generate_id()
                        run.set_extra_record_data(wandb_run_id=wandb_run_id)
                        wandb_logger = WandbLogger(save_dir=run.proj_dir,
                                                   name=run.name,  # display name for the run
                                                   # version=job_type,
                                                   # The name of the project to which this run will belong:
                                                   project=osp.split(run.proj_dir)[1],
                                                   group=run.exp_name,  # use exp_name to group runs
                                                   job_type=run.name.split('_')[-1],
                                                   id=wandb_run_id
                                                   )

                    logger_instances.append(wandb_logger)

                    # Log additional config parameters
                    # # add one parameter
                    # wandb_logger.experiment.config["key"] = value
                    #
                    # # add multiple parameters
                    # wandb_logger.experiment.config.update({key1: val1, key2: val2})
                    #
                    # # use directly wandb module
                    # wandb.config["key"] = value
                    # wandb.config.update()

        return logger_instances

    def setup_wrappers(self):
        self.task_wrapper = self.task_wrapper_getter()
        self.data_wrapper = self.data_wrapper_getter()

    def setup_trainer(self, logger_arg: str, run: Run):
        """
        Instantiate the trainer(s) using the getters.
        :param logger_arg:
        :param run:
        :return:
        """
        # TODO 向 _setup_logger 传入要记录的超参数
        logger = Config._setup_logger(logger_arg=logger_arg, run=run)
        job_type = run.job_type
        if job_type == "fit":
            self.fit_trainer = self.fit_trainer_getter(logger) if self.fit_trainer_getter \
                else self.default_trainer_getter(logger)
        elif job_type == "validate":
            self.validate_trainer = self.validate_trainer_getter(logger) if self.validate_trainer_getter \
                else self.default_trainer_getter(logger)
        elif job_type == "test":
            self.test_trainer = self.test_trainer_getter(logger) if self.test_trainer_getter \
                else self.default_trainer_getter(logger)
        elif job_type == "predict":
            self.predict_trainer = self.predict_trainer_getter(logger) if self.predict_trainer_getter \
                else self.default_trainer_getter(logger)

    @staticmethod
    def _ensure_dir_exist(dir_path, as_python_package=True):
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
            if as_python_package:
                f = open(osp.join(dir_path, "__init__.py"), 'w')  # make it become a Python package
                f.close()

    @staticmethod
    def _copy_file_from_getter(getter_func, dst_dir):
        if getter_func:
            original_file_path = inspect.getfile(getter_func)  # get the file path to the getter function
            shutil.copyfile(original_file_path, osp.join(dst_dir, osp.split(original_file_path)[1]))

    def archive_config(self, config_getter: Callable, save_dir: str, compress=True):
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        Config._copy_file_from_getter(config_getter, save_dir)

        components_dir = osp.join(save_dir, "components")
        Config._ensure_dir_exist(components_dir)

        task_wrappers_dir = osp.join(components_dir, "task_wrappers")
        Config._ensure_dir_exist(task_wrappers_dir)
        Config._copy_file_from_getter(self.task_wrapper_getter, task_wrappers_dir)

        # Get the module that defines the task_wrapper_getter
        module = inspect.getmodule(self.task_wrapper_getter)
        if hasattr(module, "get_model_instance"):  # only proceed when the model is defined from model config
            models_dir = osp.join(components_dir, "models")
            Config._ensure_dir_exist(models_dir)
            Config._copy_file_from_getter(module.get_model_instance, models_dir)

        data_wrappers_dir = osp.join(components_dir, "data_wrappers")
        Config._ensure_dir_exist(data_wrappers_dir)
        Config._copy_file_from_getter(self.data_wrapper_getter, data_wrappers_dir)

        trainers_dir = osp.join(components_dir, "trainers")
        Config._ensure_dir_exist(trainers_dir)
        Config._copy_file_from_getter(self.default_trainer_getter, trainers_dir)
        Config._copy_file_from_getter(self.fit_trainer_getter, trainers_dir)
        Config._copy_file_from_getter(self.validate_trainer_getter, trainers_dir)
        Config._copy_file_from_getter(self.test_trainer_getter, trainers_dir)
        Config._copy_file_from_getter(self.predict_trainer_getter, trainers_dir)

        if compress:
            shutil.make_archive(base_name=osp.join(osp.dirname(save_dir), osp.split(save_dir)[1]),
                                format="zip", root_dir=save_dir)
            shutil.rmtree(save_dir)

    def get_hparams(self) -> dict:
        pass

    def collect_hyperparameters(self, job_type: str) -> dict:
        """
        自动收集超参数以供记录
        :param job_type:
        :return:
        """
        if self.task_wrapper:
            all_init_args: inspect.Signature = inspect.signature(self.task_wrapper.__class__.__init__)
            filtered_init_args = {}

            for key in all_init_args.parameters.keys():
                if key == "self":
                    continue
                filtered_init_args[key] = getattr(self, key)

        if self.data_wrapper:
            pass

        # trainer
