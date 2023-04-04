import os
import os.path as osp
import sys
from typing import Optional, Callable, Union, Iterable, List, Literal
import inspect
import shutil
import re
from collections import OrderedDict
from contextlib import contextmanager
import traceback
import ast
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import Logger, CSVLogger, TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb

from entities import Run


class RootConfig:
    # Directly supported loggers
    LOGGERS = ("CSV", "TensorBoard", "wandb")

    # Names of the getters, corresponding to the function names in the config files.
    CONFIG_GETTER_NAME = "get_root_config_instance"
    MODEL_GETTER_NAME = "get_model_instance"
    TASK_WRAPPER_GETTER_NAME = "get_task_wrapper_instance"
    DATA_WRAPPER_GETTER_NAME = "get_data_wrapper_instance"
    TRAINER_GETTER_NAME = "get_trainer_instance"

    def __init__(self,
                 *,  # Compulsory keyword arguments, for better readability in config files.
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
            raise ValueError("The trainer getters can't be all None.")

        # ============= Attributes for capturing hparams =============
        # These three attributes should be defined before `_check_getter` assigning them
        self._task_wrapper_cls = None
        self._data_wrapper_cls = None
        self._trainer_cls = None

        # A "flat" dict used to temporarily store the produced local variables
        # during instantiating the task_wrapper, data_wrapper and trainer.
        # Its keys are the memory addresses of the objects,
        # while its values are dicts containing the corresponding local variables' names and values.
        self._init_local_vars = {}

        # A "structured" (nested) dict containing the hparams needed to be logged.
        self._hparams = OrderedDict()
        self._cur_run_job_type: Literal["fit", "validate", "test", "predict"] = "fit"
        # ============================================================

        # ======================== Getters ========================
        self._check_getter(task_wrapper_getter, RootConfig.TASK_WRAPPER_GETTER_NAME)
        self.task_wrapper_getter = task_wrapper_getter

        self._check_getter(data_wrapper_getter, RootConfig.DATA_WRAPPER_GETTER_NAME)
        self.data_wrapper_getter = data_wrapper_getter

        self._check_getter(default_trainer_getter, RootConfig.TRAINER_GETTER_NAME)
        self.default_trainer_getter = default_trainer_getter

        self._check_getter(fit_trainer_getter, RootConfig.TRAINER_GETTER_NAME)
        self.fit_trainer_getter = fit_trainer_getter

        self._check_getter(validate_trainer_getter, RootConfig.TRAINER_GETTER_NAME)
        self.validate_trainer_getter = validate_trainer_getter

        self._check_getter(test_trainer_getter, RootConfig.TRAINER_GETTER_NAME)
        self.test_trainer_getter = test_trainer_getter

        self._check_getter(predict_trainer_getter, RootConfig.TRAINER_GETTER_NAME)
        self.predict_trainer_getter = predict_trainer_getter
        # ============================================================

        # ======================== Core Attributes  ========================
        self.task_wrapper: Optional[pl.LightningModule] = None
        self.data_wrapper: Optional[pl.LightningDataModule] = None
        self.fit_trainer: Optional[pl.Trainer] = None
        self.validate_trainer: Optional[pl.Trainer] = None
        self.test_trainer: Optional[pl.Trainer] = None
        self.predict_trainer: Optional[pl.Trainer] = None
        self.tuner = tuner
        # Organize the trainers into a dict, convenient for enumeration and extension.
        self._trainers = OrderedDict({
            "default": OrderedDict({
                "getter": self.default_trainer_getter,
            }),
            "fit": OrderedDict({
                "getter": self.fit_trainer_getter,
                "obj": self.fit_trainer
            }),
            "validate": OrderedDict({
                "getter": self.validate_trainer_getter,
                "obj": self.validate_trainer
            }),
            "test": OrderedDict({
                "getter": self.test_trainer_getter,
                "obj": self.test_trainer
            }),
            "predict": OrderedDict({
                "getter": self.predict_trainer_getter,
                "obj": self.predict_trainer
            }),
        })
        # ===================================================================

    def _check_getter(self, getter_func: Callable, getter_name: str):
        """
        Ensure the name of a getter is correct, and its signature contains return type hints.
        """
        if getter_func:
            if getter_func.__name__ != getter_name:
                raise ValueError(f"The name of the getter is not correct, which should be \"{getter_name}\", "
                                 f"rather than \"{getter_func.__name__}\"")
            ret_type = inspect.signature(getter_func).return_annotation
            if ret_type is inspect.Signature.empty:
                raise ValueError(f"The Return type of the getter {getter_name} should be specified, "
                                 f"like `def {getter_name}() -> ReturnType:`")
            else:
                # Get the class type from the return type hints
                if getter_name == RootConfig.TASK_WRAPPER_GETTER_NAME:
                    self._task_wrapper_cls = ret_type
                elif getter_name == RootConfig.DATA_WRAPPER_GETTER_NAME:
                    self._data_wrapper_cls = ret_type
                elif getter_name == RootConfig.TRAINER_GETTER_NAME:
                    self._trainer_cls = ret_type

    # =============================== Methods for Tracking Hyper Parameters ===============================
    @contextmanager
    def _collect_frame_locals(self, collect_type: Literal["task_wrapper", "data_wrapper", "trainer"]):
        """
        A context manager to wrap some code with `sys.profile`.
        The specific collect function is specified by the argument `collect_type`.
        """
        collect_fn = None
        if collect_type == "task_wrapper":
            collect_fn = self._collect_task_wrapper_frame_locals
        elif collect_type == "data_wrapper":
            collect_fn = self._collect_data_wrapper_frame_locals
        elif collect_type == "trainer":
            collect_fn = self._collect_trainer_frame_locals

        # There may be exception during the collection process, especially when instantiating objects.
        # In this situation, the `frame.f_back` will be `None`, leading to
        # "AttributeError: 'NoneType' object has no attribute 'f_code'"  raised in the collection function.
        # The original error message will be blocked.
        # Hence, try-except is used to catch any exception, and `traceback` is used to output the original traceback.
        try:
            sys.setprofile(collect_fn)
            yield  # separate `__enter__` and `__exit__`
        except BaseException:
            print(traceback.format_exc())
            exit(1)
        finally:
            sys.setprofile(None)

    def _collect_task_wrapper_frame_locals(self, frame, event, _):
        """
        The callback function for `sys.setprofile`, used to collect local variables when initializing task wrapper.
        """
        # Only caring about the function "return" events
        if event != "return":
            return
        # Use the caller's function name to filter out underlying processes
        if frame.f_back.f_code.co_name not in (RootConfig.TASK_WRAPPER_GETTER_NAME, RootConfig.MODEL_GETTER_NAME):
            return

        f_locals = frame.f_locals  # the local variables seen by the current stack frame, a dict
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals
            # The lowermost frame corresponds to the outermost object, i.e. a task wrapper object
            if type(f_locals["self"]) == self._task_wrapper_cls:
                self._parse_frame_locals_into_hparams(f_locals, part_key="task_wrapper")

    def _collect_data_wrapper_frame_locals(self, frame, event, _):
        """ Similar to `_collect_task_wrapper_frame_locals`. """
        if event != "return":
            return
        if frame.f_back.f_code.co_name != RootConfig.DATA_WRAPPER_GETTER_NAME:
            return

        f_locals = frame.f_locals
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals
            if type(f_locals["self"]) == self._data_wrapper_cls:
                self._parse_frame_locals_into_hparams(f_locals, part_key="data_wrapper")

    def _collect_trainer_frame_locals(self, frame, event, _):
        """ Similar to `_collect_task_wrapper_frame_locals`. """
        if event != "return":
            return
        # Filter condition is special for pl.Trainer, as its `__init__` is wrapped by "_defaults_from_env_vars",
        # which is defined in "pytorch_lightning/utilities/argparse.py".
        # "insert_env_defaults" is the wrapped function's name of the decorator `_defaults_from_env_vars`.
        if frame.f_back.f_code.co_name not in ("insert_env_defaults", RootConfig.TRAINER_GETTER_NAME):
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
            - Tuple, set and other Iterable variables will be converted to list.
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
                    new_dst = []  # use list to store Iterable data
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
    def _add_hparams_to_logger(loggers: str, hparams: dict):
        """
        Some loggers support for logging hyper-parameters, call their APIS here.
        """
        if "wandb" in loggers:
            wandb.config.update(hparams)

    # ======================================================================================================

    # ====================================== Methods for Setting up ======================================
    @staticmethod
    def _setup_logger(logger_arg: str, run: Run):
        loggers, logger_instances = [], []
        for logger_str in re.split(r"[\s,]+", logger_arg):
            logger_name = logger_str.lower()
            loggers.append(logger_name)
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
                    # Call `wandb.finish()` before instantiating `WandbLogger` to avoid reusing the wandb run if there
                    # has been already created wandb run in progress, as indicated by wandb 's UserWarning.
                    wandb.finish()
                    if run.is_resuming:
                        wandb_logger = WandbLogger(save_dir=run.proj_dir,
                                                   name=run.name,  # display name for the run
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
                                                   # The name of the project to which this run will belong:
                                                   project=osp.split(run.proj_dir)[1],
                                                   group=run.exp_name,  # use exp_name to group runs
                                                   job_type=run.name.split('_')[-1],
                                                   id=wandb_run_id
                                                   )
                    logger_instances.append(wandb_logger)

        return loggers, logger_instances

    def setup_wrappers(self):
        """ Instantiate the task/data wrapper and capture their `__init__`'s arguments. """
        with self._collect_frame_locals("task_wrapper"):
            self.task_wrapper = self.task_wrapper_getter()

        with self._collect_frame_locals("data_wrapper"):
            self.data_wrapper = self.data_wrapper_getter()

    def setup_trainer(self, logger_arg: str, run: Run):
        """
        Instantiate the trainer(s) using the getters.
        :param logger_arg:
        :param run:
        :return:
        """
        loggers, logger_instances = RootConfig._setup_logger(logger_arg=logger_arg, run=run)
        self._cur_run_job_type = job_type = run.job_type

        with self._collect_frame_locals("trainer"):
            if job_type == "fit":
                self.fit_trainer = self.fit_trainer_getter(logger_instances) if self.fit_trainer_getter \
                    else self.default_trainer_getter(logger_instances)
            elif job_type == "validate":
                self.validate_trainer = self.validate_trainer_getter(logger_instances) if self.validate_trainer_getter \
                    else self.default_trainer_getter(logger_instances)
            elif job_type == "test":
                self.test_trainer = self.test_trainer_getter(logger_instances) if self.test_trainer_getter \
                    else self.default_trainer_getter(logger_instances)
            elif job_type == "predict":
                self.predict_trainer = self.predict_trainer_getter(logger_instances) if self.predict_trainer_getter \
                    else self.default_trainer_getter(logger_instances)

        RootConfig._add_hparams_to_logger(loggers, self._hparams)
        Run.save_hparams(run.run_dir, self._hparams)
        self._hparams.clear()  # clear self._hparams after saved

    # =====================================================================================================

    # ================================ Methods for Archiving Config Files ================================
    # ========== Manner 1: Save all the config files into a directory (and compress it into a zip),
    # keeping the directory structure.
    @staticmethod
    def _ensure_dir_exist(dir_path, as_python_package=True):
        if not osp.exists(dir_path):
            os.mkdir(dir_path)
            if as_python_package:
                f = open(osp.join(dir_path, "__init__.py"), 'w')  # make it become a Python package
                f.close()

    @staticmethod
    @rank_zero_only
    def _copy_file_from_getter(getter_func, dst_dir):
        if getter_func:
            original_file_path = inspect.getfile(getter_func)  # get the file path to the getter function
            shutil.copyfile(original_file_path, osp.join(dst_dir, osp.split(original_file_path)[1]))

    @rank_zero_only
    def archive_config_into_dir_or_zip(self, root_config_getter: Callable, archived_configs_dir: str,
                                       start_run_id: int, end_start_id: int,
                                       compress=True):
        archived_config_dirname = f"archived_config_run_{start_run_id}" if start_run_id == end_start_id \
            else f"archived_config_run_{start_run_id}-{end_start_id}"

        save_dir = osp.join(archived_configs_dir, archived_config_dirname)
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        RootConfig._copy_file_from_getter(root_config_getter, save_dir)

        components_dir = osp.join(save_dir, "components")
        RootConfig._ensure_dir_exist(components_dir)

        # Task wrapper
        task_wrappers_dir = osp.join(components_dir, "task_wrappers")
        RootConfig._ensure_dir_exist(task_wrappers_dir)
        RootConfig._copy_file_from_getter(self.task_wrapper_getter, task_wrappers_dir)

        # Get the module that defines the task_wrapper_getter
        module = inspect.getmodule(self.task_wrapper_getter)
        if hasattr(module, RootConfig.MODEL_GETTER_NAME):  # only proceed when the model is defined from model config
            models_dir = osp.join(components_dir, "models")
            RootConfig._ensure_dir_exist(models_dir)
            RootConfig._copy_file_from_getter(getattr(module, RootConfig.MODEL_GETTER_NAME), models_dir)

        # Data wrapper
        data_wrappers_dir = osp.join(components_dir, "data_wrappers")
        RootConfig._ensure_dir_exist(data_wrappers_dir)
        RootConfig._copy_file_from_getter(self.data_wrapper_getter, data_wrappers_dir)

        # Trainer(s)
        trainers_dir = osp.join(components_dir, "trainers")
        RootConfig._ensure_dir_exist(trainers_dir)
        for trainer in self._trainers.values():
            RootConfig._copy_file_from_getter(trainer["getter"], trainers_dir)

        if compress:
            shutil.make_archive(base_name=osp.join(osp.dirname(save_dir), osp.split(save_dir)[1]),
                                format="zip", root_dir=save_dir)
            shutil.rmtree(save_dir)

    # ========== Manner 2: Merge the configs into single file.
    @staticmethod
    def _get_import_statements_from_getter(getter_func, excludes) -> List[str]:
        """
        Return the filtered import statements from getter's source code.
        """
        if getter_func is None:
            return []

        import_statements = []
        with open(inspect.getsourcefile(getter_func)) as f:
            tree = ast.parse(f.read())
            f.seek(0)
            code_lines = f.readlines()
            # Insert a meaningless element at the index 0, because `ast.AST`'s `lineno` is 1-indexed.
            # Ref: https://docs.python.org/3/library/ast.html#ast.AST.end_col_offset
            code_lines.insert(0, '')

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_src = ''.join(code_lines[node.lineno:node.end_lineno + 1])

                # If the current import statement contains any of the keywords indicated by `excludes`,
                # it will be excluded.
                excluded = False
                for exc in excludes:
                    if import_src.find(exc) != -1:
                        excluded = True
                        break
                if not excluded:
                    import_statements.append(import_src)

        return import_statements

    @staticmethod
    def _adapt_trainer_src(trainer_getter_func, root_config_src,
                           kind: Literal["default", "fit", "validate", "test", "predict"]):
        """
        Adapt the code of trainer getter, wrapping it to a class's static method.
        And adapt the code of root config correspondingly.
        """
        if trainer_getter_func is None:
            return None, root_config_src

        trainer_src = "class {cls_name}:\n\t@staticmethod\n{src}"
        src = inspect.getsource(trainer_getter_func)
        src = '\n'.join(['\t' + line for line in src.split('\n')])  # add tabs
        cls_name = f"{kind.capitalize()}TrainerGetter"
        trainer_src = trainer_src.format(cls_name=cls_name, src=src).expandtabs(tabsize=4)
        return trainer_src, root_config_src.replace(f"{kind}_trainer_getter={RootConfig.TRAINER_GETTER_NAME}",
                                                    f"{kind}_trainer_getter={cls_name}."
                                                    f"{RootConfig.TRAINER_GETTER_NAME}")

    def _check_configs_from_same_file(self, root_config_getter) -> bool:
        """ Check whether the configs are from the same source file. """
        root_config_src_path = inspect.getsourcefile(root_config_getter)
        if inspect.getsourcefile(self.task_wrapper_getter) != root_config_src_path:
            return False
        if inspect.getsourcefile(self.data_wrapper_getter) != root_config_src_path:
            return False
        for trainer in self._trainers.values():
            if trainer["getter"] and inspect.getsourcefile(trainer["getter"]) != root_config_src_path:
                return False

        return True

    @rank_zero_only
    def archive_config_into_single(self, root_config_getter: Callable, archived_configs_dir: str,
                                   start_run_id: int, end_start_id: int):
        """ Save archived config to single file. """
        archived_config_filename = f"archived_config_run_{start_run_id}.py" if start_run_id == end_start_id \
            else f"archived_config_run_{start_run_id}-{end_start_id}.py"

        # If all the configs are from the same file, merging processing is no more needed.
        if self._check_configs_from_same_file(root_config_getter):
            shutil.copyfile(inspect.getsourcefile(root_config_getter),
                            osp.join(archived_configs_dir, archived_config_filename))
        else:
            get_import_statements = partial(RootConfig._get_import_statements_from_getter,
                                            excludes=(RootConfig.MODEL_GETTER_NAME,
                                                      RootConfig.TASK_WRAPPER_GETTER_NAME,
                                                      RootConfig.DATA_WRAPPER_GETTER_NAME,
                                                      RootConfig.TRAINER_GETTER_NAME))
            # Get the module that defines the task_wrapper_getter
            task_wrapper_module = inspect.getmodule(self.task_wrapper_getter)
            if hasattr(task_wrapper_module, RootConfig.MODEL_GETTER_NAME):
                model_getter = getattr(task_wrapper_module, RootConfig.MODEL_GETTER_NAME)
            else:
                model_getter = None

            all_import_statements = []
            with open(osp.join(archived_configs_dir, archived_config_filename), 'w') as f:
                # =========== Import Statements ===========
                all_import_statements.extend(get_import_statements(model_getter))

                all_import_statements.extend(get_import_statements(self.task_wrapper_getter))
                all_import_statements.extend(get_import_statements(self.data_wrapper_getter))

                for trainer in self._trainers.values():
                    all_import_statements.extend(get_import_statements(trainer["getter"]))

                all_import_statements.extend(get_import_statements(root_config_getter))
                all_import_statements = set(all_import_statements)  # avoid duplicate imports
                f.writelines(all_import_statements)
                f.writelines(['\n', '\n'])

                # =========== Source Code ===========
                if model_getter:
                    f.write(inspect.getsource(model_getter) + "\n\n")

                f.write(inspect.getsource(self.task_wrapper_getter) + "\n\n")
                f.write(inspect.getsource(self.data_wrapper_getter) + "\n\n")

                root_config_src = inspect.getsource(root_config_getter)
                # Adapt the source code of trainer(s) to solve name conflicts when specifying multiple trainers.
                for kind, trainer in self._trainers.items():
                    if trainer["getter"]:
                        trainer_src, root_config_src = RootConfig._adapt_trainer_src(trainer["getter"],
                                                                                     root_config_src,
                                                                                     kind)
                        f.write(trainer_src + "\n\n")
                f.write(root_config_src + '\n')
    # ======================================================================================================
