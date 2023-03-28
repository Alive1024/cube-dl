import os
import os.path as osp
import sys
from typing import Optional, Callable, Union, Iterable, List, Literal
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
    # Names of the getters, corresponding to the function names in the config files.
    TASK_WRAPPER_GETTER_NAME = "get_task_wrapper_instance"
    MODEL_GETTER_NAME = "get_model_instance"
    DATA_WRAPPER_GETTER_NAME = "get_data_wrapper_instance"
    TRAINER_GETTER_NAME = "get_trainer_instance"

    def __init__(self, *,
                 task_wrapper_getter: Callable[[], pl.LightningModule],
                 data_wrapper_getter: Callable[[], pl.LightningDataModule],
                 default_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 fit_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 validate_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 test_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 predict_trainer_getter: Optional[Callable[[Logger], pl.Trainer]] = None,
                 tuner: Tuner = None,
                 ):
        if (not default_trainer_getter) and (not fit_trainer_getter) and (not validate_trainer_getter) \
                and (not test_trainer_getter) and (not predict_trainer_getter):
            raise RuntimeError("The trainer getters can't be all None.")

        # ################## Attributes for capturing hparams ##################
        # These three attributes should be defined before `_check_getter` assigning them
        self._task_wrapper_cls = None
        self._data_wrapper_cls = None
        self._trainer_cls = None

        # A "flat" dict used to temporarily store the produced local variables
        # during instantiating the task_wrapper, data_wrapper and trainer.
        # Its keys are the memory addresses of the objects,
        # while its values are dicts containing the corresponding local variables' names and values.
        self._init_local_vars = {}

        # A "structured" (nested) dict containing the hparams
        self._hparams = OrderedDict()
        self._cur_run_job_type: Literal["fit", "validate", "test", "predict"] = "fit"
        # ######################################################################

        # ############################### Getters ###############################
        self._check_getter(task_wrapper_getter, Config.TASK_WRAPPER_GETTER_NAME)
        self.task_wrapper_getter = task_wrapper_getter

        self._check_getter(data_wrapper_getter, Config.DATA_WRAPPER_GETTER_NAME)
        self.data_wrapper_getter = data_wrapper_getter

        self._check_getter(default_trainer_getter, Config.TRAINER_GETTER_NAME)
        self.default_trainer_getter = default_trainer_getter

        self._check_getter(fit_trainer_getter, Config.TRAINER_GETTER_NAME)
        self.fit_trainer_getter = fit_trainer_getter

        self._check_getter(validate_trainer_getter, Config.TRAINER_GETTER_NAME)
        self.validate_trainer_getter = validate_trainer_getter

        self._check_getter(test_trainer_getter, Config.TRAINER_GETTER_NAME)
        self.test_trainer_getter = test_trainer_getter

        self._check_getter(predict_trainer_getter, Config.TRAINER_GETTER_NAME)
        self.predict_trainer_getter = predict_trainer_getter
        # ######################################################################

        # ########################## Core Attributes  ##########################
        self.task_wrapper: Optional[pl.LightningModule] = None
        self.data_wrapper: Optional[pl.LightningDataModule] = None
        self.fit_trainer: Optional[pl.Trainer] = None
        self.validate_trainer: Optional[pl.Trainer] = None
        self.test_trainer: Optional[pl.Trainer] = None
        self.predict_trainer: Optional[pl.Trainer] = None
        self.tuner = tuner
        # ######################################################################

    def _check_getter(self, getter: Callable, getter_name: str):
        """
        Ensure the name of a getter is correct, and its signature contains return type hints.
        """
        if getter:
            if getter.__name__ != getter_name:
                raise RuntimeError(f"The name of the getter is not correct, which should be \"{getter_name}\", "
                                   f"rather than \"{getter.__name__}\"")
            ret_type = inspect.signature(getter).return_annotation
            if ret_type is inspect.Signature.empty:
                raise RuntimeError(f"The Return type of the getter {getter_name} should be specified, "
                                   f"like `def {getter_name}() -> ReturnType:`")
            else:
                # Get the class type from the return type hints
                if getter_name == Config.TASK_WRAPPER_GETTER_NAME:
                    self._task_wrapper_cls = ret_type
                elif getter_name == Config.DATA_WRAPPER_GETTER_NAME:
                    self._data_wrapper_cls = ret_type
                elif getter_name == Config.TRAINER_GETTER_NAME:
                    self._trainer_cls = ret_type

    def _collect_task_wrapper_frame_locals(self, frame, event, arg):
        """
        The callback function for `sys.setprofile`, used to collect local variables when initializing task wrapper.
        """
        # Only caring about the function "return" events
        if event != "return":
            return
        # Use the caller's function name to filter out underlying processes
        if frame.f_back.f_code.co_name not in (Config.TASK_WRAPPER_GETTER_NAME, Config.MODEL_GETTER_NAME):
            return

        f_locals = frame.f_locals  # the local variables seen by the current stack frame, a dict
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals
            # The last frame corresponds to the outermost object, i.e. a task wrapper object
            if type(f_locals["self"]) == self._task_wrapper_cls:
                self._parse_frame_locals_into_hparams(f_locals, part_key="task_wrapper")

    def _collect_data_wrapper_frame_locals(self, frame, event, arg):
        """ Similar to `_collect_task_wrapper_frame_locals`. """
        if event != "return":
            return
        if frame.f_back.f_code.co_name != Config.DATA_WRAPPER_GETTER_NAME:
            return

        f_locals = frame.f_locals
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals
            if type(f_locals["self"]) == self._data_wrapper_cls:
                self._parse_frame_locals_into_hparams(f_locals, part_key="data_wrapper")

    def _collect_trainer_frame_locals(self, frame, event, arg):
        """ Similar to `_collect_task_wrapper_frame_locals`. """
        if event != "return":
            return
        # Filter condition is special for pl.Trainer, as its `__init__` is wrapped by "_defaults_from_env_vars",
        # which is defined in "pytorch_lightning/utilities/argparse.py".
        # "insert_env_defaults" is the wrapped function's name of the decorator `_defaults_from_env_vars`.
        if frame.f_back.f_code.co_name not in ("insert_env_defaults", Config.TRAINER_GETTER_NAME):
            return

        f_locals = frame.f_locals
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals
            # When `insert_env_defaults` or pl.Trainer's `__init__` returns,
            # the type of "self" are both `self._trainer_cls`, but only the frame
            # corresponding to `__init__` contains the actual local variables we want.
            # Hence, there are two conditions needed to be met.
            if type(f_locals["self"]) == self._trainer_cls and frame.f_code.co_name == "__init__":
                self._parse_frame_locals_into_hparams(f_locals, part_key=self._cur_run_job_type + "_trainer")

    def _parse_frame_locals_into_hparams(self, locals_dict: dict,
                                         part_key: Literal["task_wrapper", "data_wrapper",
                                                           "fit_trainer", "validate_trainer",
                                                           "test_trainer", "predict_trainer"]):
        """
        Known limitations:
            - Tuple, set and other Iterable variables will be converted to lists.
            - Generator: Values of generator type cannot be recorded,
                e.g. the return value of `torch.nn.Module.parameters`.
        """
        def __parse_obj(key, value, dst: Union[OrderedDict, List]):
            """ Parsing general objects, used in `_parse_fn`. """
            if id(value) in self._init_local_vars:
                new_dst = OrderedDict({
                    "type": str(value.__class__),  # class type
                    "args": OrderedDict()
                })
                if isinstance(dst, OrderedDict):
                    dst[key] = new_dst
                elif isinstance(dst, List):
                    dst.append(new_dst)
                _parse_fn(key, self._init_local_vars[id(value)], new_dst["args"])

        def _parse_fn(key, value, dst: Union[OrderedDict, List], exclusive_keys: Iterable = ("self",)):
            # Get rid of specific key(s), "self" must be excluded, otherwise infinite recurse will happen.
            if key in exclusive_keys:
                return

            # Atomic data types
            if (value is None) or (isinstance(value, (bool, int, float, complex, str))):
                if isinstance(dst, OrderedDict):
                    dst[key] = value
                elif isinstance(dst, List):
                    dst.append(value)

            # Iterable data types
            # Special for dict.
            elif isinstance(value, dict):
                for k, v in value.items():
                    _parse_fn(k, v, dst)
            # List, tuple, set and any other class's objects which has implemented the `__iter__` method
            elif isinstance(value, Iterable):
                # Check whether the value is "really" Iterable at first.
                # Some class (e.g. torchmetrics 's metric classes) may override the `__iter__` method
                # (which means it is `Iterable`), but do not really implement it,
                # only raise a `NotImplementedError` when trying to enumerate it.
                really_iterable = True
                try:
                    for _ in value:
                        break
                except NotImplementedError:
                    really_iterable = False

                if really_iterable:
                    new_dst = []    # use list to store Iterable data
                    if isinstance(dst, OrderedDict):
                        dst[key] = new_dst
                    elif isinstance(dst, List):
                        dst.append(new_dst)

                    for idx, v in enumerate(value):
                        _parse_fn(idx, v, new_dst)
                else:
                    __parse_obj(key, value, dst)

            # Callable ? (Classes implemented __call__ also belongs to Callable)
            # General objects
            else:
                __parse_obj(key, value, dst)

        # Delete those local variables which are not the args of `__init__`
        for local_vars in self._init_local_vars.values():
            init_param_names = inspect.signature(local_vars["self"].__class__.__init__).parameters.keys()
            useless_var_names = set(local_vars.keys()) - set(init_param_names)
            for name in useless_var_names:
                del local_vars[name]

        self._hparams[part_key] = OrderedDict({
            "type": str(locals_dict["self"].__class__),
            "args": OrderedDict()
        })
        # Construct the hparams dict recursively
        _parse_fn(None, locals_dict, self._hparams[part_key]["args"])

        # Clear it as parsing completes
        self._init_local_vars.clear()

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
        """ Instantiate the task/data wrapper and capture their `__init__`'s arguments. """
        sys.setprofile(self._collect_task_wrapper_frame_locals)
        self.task_wrapper = self.task_wrapper_getter()
        sys.setprofile(None)

        sys.setprofile(self._collect_data_wrapper_frame_locals)
        self.data_wrapper = self.data_wrapper_getter()
        sys.setprofile(None)

    def setup_trainer(self, logger_arg: str, run: Run):
        """
        Instantiate the trainer(s) using the getters.
        :param logger_arg:
        :param run:
        :return:
        """
        logger = Config._setup_logger(logger_arg=logger_arg, run=run)
        self._cur_run_job_type = job_type = run.job_type

        if job_type == "fit":
            sys.setprofile(self._collect_trainer_frame_locals)
            self.fit_trainer = self.fit_trainer_getter(logger) if self.fit_trainer_getter \
                else self.default_trainer_getter(logger)
            sys.setprofile(None)
        elif job_type == "validate":
            sys.setprofile(self._collect_trainer_frame_locals)
            self.validate_trainer = self.validate_trainer_getter(logger) if self.validate_trainer_getter \
                else self.default_trainer_getter(logger)
            sys.setprofile(None)
        elif job_type == "test":
            sys.setprofile(self._collect_trainer_frame_locals)
            self.test_trainer = self.test_trainer_getter(logger) if self.test_trainer_getter \
                else self.default_trainer_getter(logger)
            sys.setprofile(None)
        elif job_type == "predict":
            sys.setprofile(self._collect_trainer_frame_locals)
            self.predict_trainer = self.predict_trainer_getter(logger) if self.predict_trainer_getter \
                else self.default_trainer_getter(logger)
            sys.setprofile(None)

        # TODO
        # 保存 self._hparams, 向 logger 传入要记录的超参数
        print(json.dumps(self._hparams, indent=2))

        # 清空 self._hparams
        self._hparams.clear()

    # ################### Methods for Archiving Config Files ###################
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
