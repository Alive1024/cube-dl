import os
import os.path as osp
import sys
from typing import Optional, Callable, Union, Iterable, List, Literal, Tuple, Dict
import inspect
import shutil
from collections import OrderedDict
from contextlib import contextmanager
import traceback
import ast
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.loggers import Logger, CSVLogger, TensorBoardLogger, WandbLogger  # noqa
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import Logger

from c3lyr import Run, EntityFSIO


class RootConfig:
    JOB_TYPES_T = Literal["fit", "resume-fit", "validate", "test", "predict", "tune"]
    TRAINER_TYPES_T = Literal["default", "fit", "validate", "test", "predict", "tune"]
    LOGGERS_T = Union[Logger, Iterable[Logger], bool]

    # Directly supported loggers
    LOGGERS = ("CSV", "TensorBoard", "wandb")

    # Names of the getters, corresponding to the function names in the config files.
    ROOT_CONFIG_GETTER_NAME = "get_root_config_instance"

    def __init__(self,
                 *,  # Compulsory keyword arguments, for better readability in config files.
                 # Cannot import and use `TaskWrapperBase` in type hints here because of circular import.
                 # task_wrapper_getter: Callable[[], TaskWrapperBase],
                 task_wrapper_getter,
                 data_wrapper_getter: Callable[[], pl.LightningDataModule],
                 default_trainer_getter: Optional[Callable[[LOGGERS_T], pl.Trainer]] = None,
                 fit_trainer_getter: Optional[Callable[[LOGGERS_T], pl.Trainer]] = None,
                 validate_trainer_getter: Optional[Callable[[LOGGERS_T], pl.Trainer]] = None,
                 test_trainer_getter: Optional[Callable[[LOGGERS_T], pl.Trainer]] = None,
                 predict_trainer_getter: Optional[Callable[[LOGGERS_T], pl.Trainer]] = None,
                 global_seed: int = 42
                 ):
        if (not default_trainer_getter) and (not fit_trainer_getter) and (not validate_trainer_getter) \
                and (not test_trainer_getter) and (not predict_trainer_getter):
            raise ValueError("The trainer getters can't be all None.")

        # >>>>>>>>>>>>> Attributes for capturing hparams >>>>>>>>>>>>>
        # A "flat" dict used to temporarily store the produced local variables during instantiating the task_wrapper,
        # data_wrapper and trainer. Its keys are the memory addresses of the objects, while its values are dicts
        # containing the corresponding local variables' names and values.
        self._init_local_vars = {}

        # A "structured" (nested) dict containing the hparams needed to be logged.
        self._hparams = OrderedDict()
        self._cur_run_job_type = None
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>> Getters >>>>>>>>>>>>>>>>>>>>>>>>
        self.model_getter = getattr(task_wrapper_getter, "model_getter_func", None)
        if self.model_getter is None:
            raise ValueError("The model getter must be provided using the decorator `task_wrapper_getter` "
                             "when defined the task wrapper, e.g. "
                             "`@task_wrapper_getter(model_getter_func=get_model_instance)...`")
        self.task_wrapper_getter = task_wrapper_getter
        self.data_wrapper_getter = data_wrapper_getter
        self.default_trainer_getter = default_trainer_getter
        self.fit_trainer_getter = fit_trainer_getter
        self.validate_trainer_getter = validate_trainer_getter
        self.test_trainer_getter = test_trainer_getter
        self.predict_trainer_getter = predict_trainer_getter
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>> Core Attributes >>>>>>>>>>>>>>>>>>>>>>>>
        self.task_wrapper: Optional[pl.LightningModule] = None
        self.data_wrapper: Optional[pl.LightningDataModule] = None
        self.fit_trainer: Optional[pl.Trainer] = None
        self.validate_trainer: Optional[pl.Trainer] = None
        self.test_trainer: Optional[pl.Trainer] = None
        self.predict_trainer: Optional[pl.Trainer] = None

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
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.global_seed = global_seed

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods for Tracking Hyper Parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    @contextmanager
    def _collect_frame_locals(self, collect_type: Literal["task_wrapper", "data_wrapper", "trainer"]):
        """
        A context manager to wrap some code with `sys.profile`.
        The specific collect function is specified by the argument `collect_type`.
        """
        collect_fn = None
        part_key = ""
        if collect_type == "task_wrapper":
            collect_fn = self._collect_task_wrapper_frame_locals
            part_key = collect_type
        elif collect_type == "data_wrapper":
            collect_fn = self._collect_data_wrapper_frame_locals
            part_key = collect_type
        elif collect_type == "trainer":
            collect_fn = self._collect_trainer_frame_locals
            part_key = self._cur_run_job_type + "_trainer"

        # There may be exception during the collection process, especially when instantiating objects.
        # In this situation, the `frame.f_back` will be `None`, leading to
        # "AttributeError: 'NoneType' object has no attribute 'f_code'"  raised in the collection function.
        # The original error message will be blocked.
        # Hence, try-except is used to catch any exception, and `traceback` is used to output the original traceback.
        try:
            sys.setprofile(collect_fn)
            yield  # separate `__enter__` and `__exit__`
            # Start parsing after collection completes
            self._parse_frame_locals_into_hparams(part_key)
        except BaseException as e:  # noqa
            print(traceback.format_exc())
            raise e
        finally:
            sys.setprofile(None)

    def _collect_task_wrapper_frame_locals(self, frame, event, _):
        """
        The callback function for `sys.setprofile`, used to collect local variables when initializing task wrapper.
        """
        # Only caring about the function "return" events.
        # Just before the function returns, there exist all local variables we want.
        if event != "return":
            return
        # Use the caller's function name to filter out underlying instantiation processes.
        # We only care about the objects instantiated in these two getter functions.
        if frame.f_back.f_code.co_name not in (self.task_wrapper_getter.__name__, self.model_getter.__name__):
            return

        f_locals = frame.f_locals  # the local variables seen by the current stack frame, a dict
        if "self" in f_locals:  # for normal objects, there must be "self"
            # Put the local variables into a dict, using the address as key. Note that the local variables include
            # both arguments of `__init__` and variables defined within `__init__`.
            self._init_local_vars[id(f_locals["self"])] = f_locals

    def _collect_data_wrapper_frame_locals(self, frame, event, _):
        """ Similar to `_collect_task_wrapper_frame_locals`. """
        if event != "return":
            return
        if frame.f_back.f_code.co_name != self.data_wrapper_getter.__name__:
            return

        f_locals = frame.f_locals
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals

    def _collect_trainer_frame_locals(self, frame, event, _):
        """ Similar to `_collect_task_wrapper_frame_locals`. """
        if event != "return":
            return

        # Stop if the trainer has been not assigned
        if getattr(self, f"{self._cur_run_job_type}_trainer_getter") is None:
            return
        else:
            # Filter condition is special for pl.Trainer, as its `__init__` is wrapped by `_defaults_from_env_vars`,
            # which is defined in "pytorch_lightning/utilities/argparse.py".
            # "insert_env_defaults" is the wrapped function's name of the decorator `_defaults_from_env_vars`.
            if frame.f_back.f_code.co_name not in ("insert_env_defaults",
                                                   getattr(self, f"{self._cur_run_job_type}_trainer_getter").__name__):
                return

        f_locals = frame.f_locals
        # When `insert_env_defaults` returns (happens after pl.Trainer's __init__ returns),
        # `f_locals` contains the same `pl.Trainer` object as pl.Trainer's __init__,
        # but the local variables are incomplete.
        # This should be filtered out, or it will override original full pl.Trainer's arguments.
        # Here, use the name of code object being executed in current frame (pl.Trainer's "__init__") to further filter.
        if "self" in f_locals and frame.f_code.co_name == "__init__":
            self._init_local_vars[id(f_locals["self"])] = f_locals

    def _parse_frame_locals_into_hparams(self, part_key: Literal["task_wrapper", "data_wrapper",
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
            if id(value) in all_init_args:
                new_dst = OrderedDict({
                    "type": str(value.__class__),  # class type
                    "args": OrderedDict()
                })
                if isinstance(dst, OrderedDict):
                    dst[key] = new_dst
                elif isinstance(dst, List):
                    dst.append(new_dst)
                _parse_fn(key, all_init_args[id(value)], new_dst["args"])

        def _parse_fn(key, value, dst: Union[OrderedDict, List], exclusive_keys: Iterable = ("self",)):
            # Get rid of specific key(s), "self" must be excluded, otherwise infinite recursion will happen.
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
                # And some variables are "empty" that cannot also be iterated (e.g. 0-d tensor), `TypeError`
                # will be raised in this case.
                really_iterable = True
                try:
                    for _ in value:
                        break
                except NotImplementedError:
                    really_iterable = False
                except TypeError:
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

        # Finding the starting point of parsing, i.e. the object we are creating
        target_obj_locals_dict = None
        for local_vars in self._init_local_vars.values():
            value_type = type(local_vars["self"])
            if part_key == "task_wrapper" and value_type == self.task_wrapper.__class__:
                target_obj_locals_dict = local_vars
                break
            elif part_key == "data_wrapper" and value_type == self.data_wrapper.__class__:
                target_obj_locals_dict = local_vars
                break
            elif part_key.find("trainer") != -1 and value_type == getattr(self, part_key).__class__:
                target_obj_locals_dict = local_vars
                break

        # Sift out the arguments of `__init__` from `self._init_local_vars`.
        # Local variables defined within `__init__` should not be included.
        all_init_args = {}
        for var_id, local_vars in self._init_local_vars.items():
            all_init_args[var_id] = {}
            # Get the names of arguments of `__init__` through its signature
            init_args_names = inspect.signature(local_vars["self"].__class__.__init__).parameters.keys()
            for arg_name in init_args_names:
                # Ignore placeholder arguments "args" and "kwargs"
                if arg_name not in ("args", "kwargs"):
                    all_init_args[var_id][arg_name] = local_vars[arg_name]

        self._hparams[part_key] = OrderedDict({
            "type": str(target_obj_locals_dict["self"].__class__),
            "args": OrderedDict()
        })
        # Construct the hparams dict recursively
        _parse_fn(None, target_obj_locals_dict, self._hparams[part_key]["args"])

        # Clear it as parsing completes
        self._init_local_vars.clear()

    @staticmethod
    def _add_hparams_to_logger(loggers: Union[dict, bool], hparams: dict):
        """
        Some loggers support for logging hyper-parameters, call their APIs here.
        """
        if loggers:
            # Executed only on rank 0, more details at: https://github.com/Lightning-AI/lightning/issues/13166
            if "wandb" in loggers and rank_zero_only.rank == 0:  # noqa
                # Note: use directly wandb module here (i.e. `wandb.config.update(hparams)`)
                # will trigger an error: "wandb.errors.Error: You must call wandb.init() before wandb.config.update"
                loggers["wandb"].experiment.config.update(hparams)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods for Setting up >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    @staticmethod
    def _setup_logger(logger_arg: Union[str, List[str]], run: Run) -> Union[Dict[str, LOGGERS_T], bool]:
        loggers = {}
        belonging_exp = run.belonging_exp
        belonging_proj = belonging_exp.belonging_proj
        if isinstance(logger_arg, str):
            logger_arg = [logger_arg]
        for logger_str in logger_arg:
            logger_name = logger_str.lower()

            if logger_name == "false":
                return False
            elif logger_name == "csv":
                # Logs are saved to `os.path.join(save_dir, name, version)`.
                logger_instance = CSVLogger(save_dir=belonging_proj.proj_dir,
                                            name=belonging_exp.dirname,
                                            version=run.dirname)
            elif logger_name == "tensorboard":
                logger_instance = TensorBoardLogger(save_dir=belonging_proj.proj_dir,
                                                    name=belonging_exp.dirname,
                                                    version=run.dirname)
            elif logger_name == "wandb":
                get_wandb_logger = partial(WandbLogger,
                                           save_dir=belonging_proj.proj_dir,
                                           name=run.dirname,  # display name for the run
                                           # The name of the project to which this run will belong:
                                           project=belonging_proj.dirname,
                                           group=belonging_exp.dirname,  # use exp_name to group runs
                                           job_type=run.job_type,
                                           id=run.global_id)
                logger_instance = get_wandb_logger(resume="must") if run.is_resuming else get_wandb_logger()
            else:
                raise ValueError("Unrecognized logger name: " + logger_str)

            loggers[logger_name] = logger_instance

        return loggers

    def setup_wrappers(self):
        """ Instantiate the task/data wrapper and capture their `__init__`'s arguments. """
        with self._collect_frame_locals("task_wrapper"):
            self.task_wrapper = self.task_wrapper_getter()

        with self._collect_frame_locals("data_wrapper"):
            self.data_wrapper = self.data_wrapper_getter()

    def setup_trainer(self, logger_arg: Union[str, List[str]], run: Run):
        """
        Instantiate the trainer(s) using the getters.
        :param logger_arg:
        :param run:
        :return:
        """
        loggers = RootConfig._setup_logger(logger_arg=logger_arg, run=run)
        self._cur_run_job_type = job_type = run.job_type
        logger_param = False if not loggers else list(loggers.values())

        with self._collect_frame_locals("trainer"):
            if job_type == "fit":
                if self.fit_trainer_getter is None:
                    self.fit_trainer_getter = self.default_trainer_getter
                self.fit_trainer = self.fit_trainer_getter(logger_param)
            elif job_type == "validate":
                if self.validate_trainer_getter is None:
                    self.validate_trainer_getter = self.default_trainer_getter
                self.validate_trainer = self.validate_trainer_getter(logger_param)
            elif job_type == "test":
                if self.test_trainer_getter is None:
                    self.test_trainer_getter = self.default_trainer_getter
                self.test_trainer = self.test_trainer_getter(logger_param)
            elif job_type == "predict":
                if self.predict_trainer_getter is None:
                    self.predict_trainer_getter = self.default_trainer_getter
                self.predict_trainer = self.predict_trainer_getter(logger_param)

        RootConfig._add_hparams_to_logger(loggers, self._hparams)
        EntityFSIO.save_hparams(run.run_dir, self._hparams, global_seed=self.global_seed)
        self._hparams.clear()  # clear self._hparams after saved

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods for Archiving Config Files >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Manner 1 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Save all the config files into a directory (and compress it into a zip),
    # keeping the directory structure.

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

    @rank_zero_only
    def archive_config_into_dir_or_zip(self, root_config_getter_func: Callable, archived_configs_dir: str,
                                       run_id: str, to_zip=True):
        archived_config_dirname = f"archived_config_run_{run_id}"

        save_dir = osp.join(archived_configs_dir, archived_config_dirname)
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        RootConfig._copy_file_from_getter(root_config_getter_func, save_dir)

        components_dir = osp.join(save_dir, "components")
        RootConfig._ensure_dir_exist(components_dir)

        # Task wrapper
        task_wrappers_dir = osp.join(components_dir, "task_wrappers")
        RootConfig._ensure_dir_exist(task_wrappers_dir)
        RootConfig._copy_file_from_getter(self.task_wrapper_getter, task_wrappers_dir)

        # Model
        models_dir = osp.join(components_dir, "models")
        RootConfig._ensure_dir_exist(models_dir)
        RootConfig._copy_file_from_getter(self.model_getter, models_dir)

        # Data wrapper
        data_wrappers_dir = osp.join(components_dir, "data_wrappers")
        RootConfig._ensure_dir_exist(data_wrappers_dir)
        RootConfig._copy_file_from_getter(self.data_wrapper_getter, data_wrappers_dir)

        # Trainer(s)
        trainers_dir = osp.join(components_dir, "trainers")
        RootConfig._ensure_dir_exist(trainers_dir)
        for trainer in self._trainers.values():
            RootConfig._copy_file_from_getter(trainer["getter"], trainers_dir)

        if to_zip:
            shutil.make_archive(base_name=osp.join(osp.dirname(save_dir), osp.split(save_dir)[1]),
                                format="zip", root_dir=save_dir)
            shutil.rmtree(save_dir)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Manner 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Merge the configs into single file.

    @staticmethod
    def _get_import_stats_global_vars(getter_func, excludes) -> Tuple[List[str], List[str]]:
        """
        Return the filtered import statements (& global variables) from getter's source code.
        """
        if getter_func is None:
            return [], []

        import_stats, global_vars = [], []
        with open(inspect.getsourcefile(getter_func)) as f:
            tree = ast.parse(f.read())  # get the abstract syntax tree of the source file
            f.seek(0)
            code_lines = f.readlines()
            # Insert a meaningless element at the index 0, because `ast.AST`'s `lineno` is 1-indexed.
            # Ref: https://docs.python.org/3/library/ast.html#ast.AST.end_col_offset
            code_lines.insert(0, '')

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Copy corresponding import lines(s) from source file
                import_src = ''.join(code_lines[node.lineno: node.end_lineno + 1])

                # If the current import statement contains any of the keywords indicated by `excludes`,
                # it will be excluded.
                excluded = False
                for exc in excludes:
                    if import_src.find(exc) != -1:
                        excluded = True
                        break
                if not excluded:
                    import_stats.append(import_src)
            # Global variables
            elif isinstance(node, ast.Assign):
                global_vars.append(''.join(code_lines[node.lineno: node.end_lineno + 1]))

        return import_stats, global_vars

    @staticmethod
    def _adapt_trainer_src(trainer_getter_func, root_config_src,
                           kind: TRAINER_TYPES_T):
        """
        Adapt the code of trainer getter, wrapping it to a class's static method.
        And adapt the code of root config correspondingly.
        """
        if trainer_getter_func is None:
            return None, root_config_src

        trainer_src = inspect.getsource(trainer_getter_func)
        trainer_src = '\n'.join(['\t' + line for line in trainer_src.split('\n')])  # add tabs
        cls_name = f"{kind.capitalize()}TrainerGetter"
        trainer_src = f"class {cls_name}:\n\t@staticmethod\n{trainer_src}".expandtabs(tabsize=4)
        return trainer_src, root_config_src.replace(f"{kind}_trainer_getter={trainer_getter_func.__name__}",
                                                    f"{kind}_trainer_getter={cls_name}."
                                                    f"{trainer_getter_func.__name__}")

    def _check_configs_from_same_file(self, root_config_getter_func) -> bool:
        """ Check whether the configs are from the same source file. """
        root_config_src_path = inspect.getsourcefile(root_config_getter_func)
        if inspect.getsourcefile(self.task_wrapper_getter) != root_config_src_path:
            return False
        if inspect.getsourcefile(self.data_wrapper_getter) != root_config_src_path:
            return False
        for trainer in self._trainers.values():
            if trainer["getter"] and inspect.getsourcefile(trainer["getter"]) != root_config_src_path:
                return False

        return True

    @rank_zero_only
    def archive_config_into_single(self, root_config_getter_func: Callable, archived_configs_dir: str, run_id: str):
        """ Save archived config to single file. """
        archived_config_filename = f"archived_config_run_{run_id}.py"

        # If all the configs are from the same file, merging processing is no more needed.
        if self._check_configs_from_same_file(root_config_getter_func):
            shutil.copyfile(inspect.getsourcefile(root_config_getter_func),
                            osp.join(archived_configs_dir, archived_config_filename))
        else:
            # The unnecessary import statements for task wrapper, data wrapper and trainer(s) should be removed.
            excludes = [self.model_getter.__name__,
                        self.task_wrapper_getter.__name__,
                        self.data_wrapper_getter.__name__]
            for trainer in self._trainers.values():
                if trainer["getter"]:
                    excludes.append(trainer["getter"].__name__)

            get_import_stats_global_vars = partial(RootConfig._get_import_stats_global_vars,
                                                   excludes=excludes)
            all_import_stats, all_global_vars = [], []
            with open(osp.join(archived_configs_dir, archived_config_filename), 'w') as f:
                # >>>>>>>>>>> Import Statements (& global variables) >>>>>>>>>>>
                # 1. Model
                import_stats, global_vars = get_import_stats_global_vars(self.model_getter)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # 2. Task wrapper
                import_stats, global_vars = get_import_stats_global_vars(self.task_wrapper_getter)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # 3. Data wrapper
                import_stats, global_vars = get_import_stats_global_vars(self.data_wrapper_getter)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # 4. Trainer(s)
                for trainer in self._trainers.values():
                    import_stats, global_vars = get_import_stats_global_vars(trainer["getter"])
                    all_import_stats.extend(import_stats)
                    all_global_vars.extend(global_vars)

                # 5. Root config
                import_stats, global_vars = get_import_stats_global_vars(root_config_getter_func)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # Process & write
                all_import_stats = set(all_import_stats)  # avoid duplicate imports
                all_global_vars = set(all_global_vars)
                f.writelines(all_import_stats)
                f.writelines(['\n', '\n'])
                if len(all_global_vars) != 0:
                    f.writelines(all_global_vars)
                    f.write('\n')
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # >>>>>>>>>>>>>>>>>>>>>> Source Code >>>>>>>>>>>>>>>>>>>>>>
                f.write(inspect.getsource(self.model_getter) + "\n\n")
                f.write(inspect.getsource(self.task_wrapper_getter) + "\n\n")
                f.write(inspect.getsource(self.data_wrapper_getter) + "\n\n")

                root_config_src = inspect.getsource(root_config_getter_func)
                # Adapt the source code of trainer(s) to solve potential name conflicts
                # when specifying multiple trainers.
                for kind, trainer in self._trainers.items():
                    if trainer["getter"]:
                        trainer_src, root_config_src = RootConfig._adapt_trainer_src(trainer["getter"],
                                                                                     root_config_src,
                                                                                     kind)  # noqa
                        f.write(trainer_src + "\n\n")
                f.write(root_config_src + '\n')
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
