import ast
import inspect
import json
import os
import os.path as osp
import re
import shutil
import sys
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from functools import partial
from typing import Literal

from cube.c3lyr import Run
from cube.callbacks.callback import CubeCallback, CubeCallbackList
from cube.core import CubeDataModule, CubeRunner, CubeTaskModule
from cube.dist_utils import rank_zero_only
from cube.types import LOGGER_GETTER_T, LOGGER_T, RUNNER_TYPES_T

ARCHIVED_CONFIG_FORMAT = Literal["single-py", "zip", "dir"]


class RootConfig:
    # The uniform name of the root config getter
    ROOT_CONFIG_GETTER_NAME = "get_root_config"

    def __init__(
        self,
        *,  # Compulsory keyword arguments, for better readability in config files.
        model_getters,
        task_module_getter: Callable[[], CubeTaskModule],
        data_module_getter: Callable[[], CubeTaskModule],
        default_runner_getter: Callable[[LOGGER_T], CubeRunner] | None = None,
        fit_runner_getter: Callable[[LOGGER_T], CubeRunner] | None = None,
        validate_runner_getter: Callable[[LOGGER_T], CubeRunner] | None = None,
        test_runner_getter: Callable[[LOGGER_T], CubeRunner] | None = None,
        predict_runner_getter: Callable[[LOGGER_T], CubeRunner] | None = None,
        logger_getters: LOGGER_GETTER_T | Iterable[LOGGER_GETTER_T] | None = None,
        seed_func: Callable,
        global_seed: int | None = 42,
        archive_hparams: bool = True,
        archive_config: ARCHIVED_CONFIG_FORMAT | bool = "single-py",
        callbacks: CubeCallback | Iterable[CubeCallback] | None = None,
    ):
        if (
            (not default_runner_getter)
            and (not fit_runner_getter)
            and (not validate_runner_getter)
            and (not test_runner_getter)
            and (not predict_runner_getter)
        ):
            raise ValueError("The runner getters cannot be all `None`.")

        # >>>>>>>>>>>>> Attributes for capturing hparams >>>>>>>>>>>>>
        # A "flat" dict used to temporarily store the produced local variables during instantiating the task_module,
        # data_module and runner(s). Its keys are the memory addresses of the objects, while its values are dicts
        # containing the corresponding local variables' names and values.
        self._init_local_vars = {}

        # A "structured" (nested) dict containing the hparams needed to be logged.
        self._hparams = OrderedDict()
        self._cur_run_job_type = None
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>> Getters >>>>>>>>>>>>>>>>>>>>>>>>
        self.model_getters = model_getters if isinstance(model_getters, Iterable) else [model_getters]
        self.task_module_getter = task_module_getter
        self.data_module_getter = data_module_getter
        self.default_runner_getter = default_runner_getter
        self.fit_runner_getter = fit_runner_getter
        self.validate_runner_getter = validate_runner_getter
        self.test_runner_getter = test_runner_getter
        self.predict_runner_getter = predict_runner_getter
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>>>>> Core Attributes >>>>>>>>>>>>>>>>>>>>>>>>
        self.task_module: CubeTaskModule | None = None
        self.data_module: CubeDataModule | None = None
        self.fit_runner: CubeRunner | None = None
        self.validate_runner: CubeRunner | None = None
        self.test_runner: CubeRunner | None = None
        self.predict_runner: CubeRunner | None = None

        # Organize the runners into a dict, convenient for enumeration and extension.
        self._runners = OrderedDict(
            {
                "default": OrderedDict(
                    {
                        "getter": self.default_runner_getter,
                    }
                ),
                "fit": OrderedDict({"getter": self.fit_runner_getter, "obj": self.fit_runner}),
                "validate": OrderedDict(
                    {
                        "getter": self.validate_runner_getter,
                        "obj": self.validate_runner,
                    }
                ),
                "test": OrderedDict({"getter": self.test_runner_getter, "obj": self.test_runner}),
                "predict": OrderedDict({"getter": self.predict_runner_getter, "obj": self.predict_runner}),
            }
        )

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        self.logger_getters = logger_getters
        seed_func(global_seed)
        self.global_seed = global_seed
        self.archive_hparams = archive_hparams
        self.archive_config = archive_config
        self.callbacks: CubeCallbackList = CubeCallbackList(callbacks)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods for Tracking Hyper Parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    @contextmanager
    def _collect_frame_locals(
        self, collect_type: Literal["task_module", "data_module", "runner"], enabled: bool = True
    ):
        """
        A context manager to wrap some code with `sys.profile`.
        The specific collect function is specified by the argument `collect_type`.
        """
        if enabled:
            if collect_type == "task_module":
                collect_fn = self._collect_task_module_frame_locals
                part_key = collect_type
            elif collect_type == "data_module":
                collect_fn = self._collect_data_module_frame_locals
                part_key = collect_type
            elif collect_type == "runner":
                collect_fn = self._collect_runner_frame_locals
                part_key = self._cur_run_job_type + "_runner"
            else:
                raise ValueError(f"Unrecognized collect type: {collect_type}")

            # There may be exception during the collection process, especially when instantiating objects. In this
            # situation, the `frame.f_back` will be `None`, leading to "AttributeError: 'NoneType' object has no
            # attribute 'f_code'"  raised in the collection function. The original error message will be blocked.
            # Hence, try-except is used to catch any exception, and `traceback` is used to output the original
            # traceback.
            try:
                sys.setprofile(collect_fn)  # pass a specific collection function as the callback
                yield  # separate `__enter__` and `__exit__`
                # Start parsing after collection completes
                self._parse_frame_locals_into_hparams(part_key)
            except BaseException as e:  # noqa
                print(traceback.format_exc())
                raise e
            finally:
                sys.setprofile(None)
        # Do nothing if not enabled
        else:
            yield

    def _collect_task_module_frame_locals(self, frame, event, _):
        """
        The callback function for `sys.setprofile`, used to collect local variables when initializing task wrapper.
        """
        # Only caring about the function "return" events.
        # Just before the function returns, there exist all local variables we want.
        if event != "return":
            return

        # Use the caller's function name to filter out underlying instantiation processes.
        # We only care about the objects instantiated in these two getter functions.
        if frame.f_back.f_code.co_name not in (
            self.task_module_getter.__name__,
            self.model_getter.__name__,
        ):
            return

        f_locals = frame.f_locals  # the local variables seen by the current stack frame, a dict
        if "self" in f_locals:  # for normal objects, there must be "self"
            # Put the local variables into a dict, using the address as key. Note that the local variables include
            # both arguments of `__init__` and variables defined within `__init__`.
            self._init_local_vars[id(f_locals["self"])] = f_locals

    def _collect_data_module_frame_locals(self, frame, event, _):
        """Similar to `_collect_task_wrapper_frame_locals`."""
        if event != "return":
            return
        if frame.f_back.f_code.co_name != self.data_module_getter.__name__:
            return

        f_locals = frame.f_locals
        if "self" in f_locals:
            self._init_local_vars[id(f_locals["self"])] = f_locals

    def _collect_runner_frame_locals(self, frame, event, _):
        """Similar to `_collect_task_wrapper_frame_locals`."""
        if event != "return":
            return

        # Stop if the runner has been not assigned
        if getattr(self, f"{self._cur_run_job_type}_runner_getter") is None:
            return
        else:
            # Filter condition is special for pl.Trainer, as its `__init__` is wrapped by `_defaults_from_env_vars`,
            # which is defined in "pytorch_lightning/utilities/argparse.py".
            # "insert_env_defaults" is the wrapped function's name of the decorator `_defaults_from_env_vars`.
            if frame.f_back.f_code.co_name not in (
                "insert_env_defaults",
                getattr(self, f"{self._cur_run_job_type}_trainer_getter").__name__,
            ):
                return

        f_locals = frame.f_locals
        # When `insert_env_defaults` returns (happens after pl.Trainer's __init__ returns),
        # `f_locals` contains the same `pl.Trainer` object as pl.Trainer's __init__,
        # but the local variables are incomplete.
        # This should be filtered out, or it will override original full pl.Trainer's arguments.
        # Here, use the name of code object being executed in current frame (pl.Trainer's "__init__") to further filter.
        if "self" in f_locals and frame.f_code.co_name == "__init__":
            self._init_local_vars[id(f_locals["self"])] = f_locals

    def _parse_frame_locals_into_hparams(  # noqa: C901
        self,
        part_key: Literal[
            "task_module",
            "data_module",
            "fit_runner",
            "validate_runner",
            "test_runner",
            "predict_runner",
        ],
    ):
        """
        Known limitations:
            - Tuple, set and other Iterable variables will be converted to list.
            - Generator: Values of generator type cannot be recorded,
                e.g. the return value of `torch.nn.Module.parameters`.
        """

        def __parse_obj(key, value, dst: OrderedDict | list):
            """Parsing general objects, used in `_parse_fn`."""
            if id(value) in all_init_args:
                new_dst = OrderedDict({"type": str(value.__class__), "args": OrderedDict()})  # class type
                if isinstance(dst, OrderedDict):
                    dst[key] = new_dst
                elif isinstance(dst, list):
                    dst.append(new_dst)
                _parse_fn(key, all_init_args[id(value)], new_dst["args"])

        def _parse_fn(  # noqa: C901
            key,
            value,
            dst: OrderedDict | list,
            exclusive_keys: Iterable = ("self",),
        ):
            # Get rid of specific key(s), "self" must be excluded, otherwise infinite recursion will happen.
            if key in exclusive_keys:
                return

            # Atomic data types
            if (value is None) or (isinstance(value, bool | int | float | complex | str)):
                if isinstance(dst, OrderedDict):
                    dst[key] = value
                elif isinstance(dst, list):
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
                    elif isinstance(dst, list):
                        dst.append(new_dst)

                    for idx, v in enumerate(value):
                        _parse_fn(idx, v, new_dst)
                else:
                    __parse_obj(key, value, dst)

            # Callable ? (Classes implemented __call__ also belongs to Callable)
            # General objects
            else:
                __parse_obj(key, value, dst)

        # Find the starting point of parsing, i.e. the object being creating
        target_obj_locals_dict = None
        for local_vars in self._init_local_vars.values():
            value_type = type(local_vars["self"])
            if (
                (part_key == "task_module" and value_type == self.task_module.__class__)
                or (part_key == "data_module" and value_type == self.data_module.__class__)
                or (part_key.find("runner") != -1 and value_type == getattr(self, part_key).__class__)
            ):
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

        self._hparams[part_key] = OrderedDict(
            {
                "type": str(target_obj_locals_dict["self"].__class__),
                "args": OrderedDict(),
            }
        )
        # Construct the hparams dict recursively
        _parse_fn(None, target_obj_locals_dict, self._hparams[part_key]["args"])

        # Clear it as parsing completes
        self._init_local_vars.clear()

    @staticmethod
    @rank_zero_only
    def save_hparams(run_dir, hparams: OrderedDict, **kwargs):
        """Save hparams to json files. "hparams.json" always indicates the latest, similar to "metrics.csv"."""
        hparams_json_path = osp.join(run_dir, "hparams.json")
        if osp.exists(hparams_json_path):
            indices = []
            for filename in os.listdir(run_dir):
                match = re.match(r"hparams_(\d+).json", filename)
                if match:
                    indices.append(int(match.group(1)))

            max_cnt = 1 if len(indices) == 0 else max(indices) + 1
            shutil.move(
                hparams_json_path,
                osp.splitext(hparams_json_path)[0] + f"_{max_cnt}.json",
            )

        hparams.update(kwargs)
        with open(hparams_json_path, "w", encoding="utf-8") as f:
            json.dump(hparams, f, indent=2)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods for Setting up >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def setup_task_data_modules(self):
        """Instantiate the task/data modules."""
        with self._collect_frame_locals("task_module", enabled=self.archive_hparams):
            self.task_module = self.task_module_getter()
        with self._collect_frame_locals("data_module", enabled=self.archive_hparams):
            self.data_module = self.data_module_getter()

    def setup_runners(self, run: Run):  # noqa: C901
        """Set up the runner(s) using the getters."""
        self._cur_run_job_type = job_type = run.job_type
        logger_param = (
            False if self.logger_getters is None else [logger_getter(run) for logger_getter in self.logger_getters]
        )

        with self._collect_frame_locals("runner", enabled=self.archive_hparams):
            if job_type == "fit":
                if self.fit_runner_getter is None:
                    self.fit_runner_getter = self.default_runner_getter
                self.fit_runner = self.fit_runner_getter(logger_param)
            elif job_type == "validate":
                if self.validate_runner_getter is None:
                    self.validate_runner_getter = self.default_runner_getter
                self.validate_runner = self.validate_runner_getter(logger_param)
            elif job_type == "test":
                if self.test_runner_getter is None:
                    self.test_runner_getter = self.default_runner_getter
                self.test_runner = self.test_runner_getter(logger_param)
            elif job_type == "predict":
                if self.predict_runner_getter is None:
                    self.predict_runner_getter = self.default_runner_getter
                self.predict_runner = self.predict_runner_getter(logger_param)

        # Save hyper-parameters
        if self.archive_hparams:
            self.save_hparams(run.run_dir, self._hparams, global_seed=self.global_seed)
            self._hparams.clear()  # clear self._hparams after saved
        # Save config file(s)
        if self.archive_config:
            self._save_config(self.archive_config, run)

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
                # Make it become a Python package
                with open(osp.join(dir_path, "__init__.py"), "w"):
                    pass

    @staticmethod
    def _copy_file_from_getter(getter_func, dst_dir):
        if getter_func:
            original_file_path = inspect.getfile(getter_func)  # get the file path to the getter function
            shutil.copyfile(original_file_path, osp.join(dst_dir, osp.split(original_file_path)[1]))

    def _archive_config_into_dir_or_zip(
        self,
        root_config_getter: Callable,
        archived_configs_dir: str,
        run_id: str,
        to_zip=True,
    ):
        archived_config_dirname = f"archived_config_run_{run_id}"

        save_dir = osp.join(archived_configs_dir, archived_config_dirname)
        if not osp.exists(save_dir):
            os.mkdir(save_dir)

        RootConfig._copy_file_from_getter(root_config_getter, save_dir)

        components_dir = osp.join(save_dir, "components")
        RootConfig._ensure_dir_exist(components_dir)

        # Task module
        task_wrappers_dir = osp.join(components_dir, "task_modules")
        RootConfig._ensure_dir_exist(task_wrappers_dir)
        RootConfig._copy_file_from_getter(self.task_module_getter, task_wrappers_dir)

        # Model(s)
        models_dir = osp.join(components_dir, "models")
        RootConfig._ensure_dir_exist(models_dir)
        for model_getter in self.model_getters:
            RootConfig._copy_file_from_getter(model_getter, models_dir)

        # Data module
        data_wrappers_dir = osp.join(components_dir, "data_modules")
        RootConfig._ensure_dir_exist(data_wrappers_dir)
        RootConfig._copy_file_from_getter(self.data_module_getter, data_wrappers_dir)

        # Runner(s)
        runners_dir = osp.join(components_dir, "runners")
        RootConfig._ensure_dir_exist(runners_dir)
        for runner in self._runners.values():
            RootConfig._copy_file_from_getter(runner["getter"], runners_dir)

        if to_zip:
            shutil.make_archive(
                base_name=osp.join(osp.dirname(save_dir), osp.split(save_dir)[1]),
                format="zip",
                root_dir=save_dir,
            )
            shutil.rmtree(save_dir)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Manner 2 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Merge the configs into single file.

    @staticmethod
    def _get_import_stats_global_vars(getter_func, excludes) -> tuple[list[str], list[str]]:
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
            code_lines.insert(0, "")

        for node in tree.body:
            if isinstance(node, ast.Import | ast.ImportFrom):
                # Copy corresponding import lines(s) from source file
                import_src = "".join(code_lines[node.lineno : node.end_lineno + 1])

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
                global_vars.append("".join(code_lines[node.lineno : node.end_lineno + 1]))

        return import_stats, global_vars

    @staticmethod
    def _adapt_runner_src(runner_getter_func, root_config_src, kind: RUNNER_TYPES_T):
        """
        Adapt the code of runner getter, wrapping it to a class's static method.
        And adapt the code of root config correspondingly.
        """
        if runner_getter_func is None:
            return None, root_config_src

        runner_src = inspect.getsource(runner_getter_func)
        runner_src = "\n".join(["\t" + line for line in runner_src.split("\n")])  # add tabs
        cls_name = f"{kind.capitalize()}RunnerGetter"
        runner_src = f"class {cls_name}:\n\t@staticmethod\n{runner_src}".expandtabs(tabsize=4)
        return runner_src, root_config_src.replace(
            f"{kind}_runner_getter={runner_getter_func.__name__}",
            f"{kind}_runner_getter={cls_name}." f"{runner_getter_func.__name__}",
        )

    def _check_configs_from_same_file(self, root_config_getter) -> bool:
        """Check whether the configs are from the same source file."""
        root_config_src_path = inspect.getsourcefile(root_config_getter)
        if inspect.getsourcefile(self.task_module_getter) != root_config_src_path:
            return False
        if inspect.getsourcefile(self.data_module_getter) != root_config_src_path:
            return False
        for runner in self._runners.values():
            if runner["getter"] and inspect.getsourcefile(runner["getter"]) != root_config_src_path:
                return False

        return True

    def _archive_config_into_single(self, root_config_getter: Callable, archived_configs_dir: str, run_id: str):
        """Save archived config to single file."""
        archived_config_filename = f"archived_config_run_{run_id}.py"

        # If all the configs are from the same file, merging processing is no more needed.
        if self._check_configs_from_same_file(root_config_getter):
            shutil.copyfile(
                inspect.getsourcefile(root_config_getter),
                osp.join(archived_configs_dir, archived_config_filename),
            )
        else:
            # The unnecessary import statements for task wrapper, data wrapper and runner(s) should be removed.
            excludes = [
                *[model_getter.__name__ for model_getter in self.model_getters],
                self.task_module_getter.__name__,
                self.data_module_getter.__name__,
            ]
            for runner in self._runners.values():
                if runner["getter"]:
                    excludes.append(runner["getter"].__name__)

            get_import_stats_global_vars = partial(RootConfig._get_import_stats_global_vars, excludes=excludes)
            all_import_stats, all_global_vars = [], []
            with open(osp.join(archived_configs_dir, archived_config_filename), "w") as f:
                # >>>>>>>>>>> Import Statements (& global variables) >>>>>>>>>>>
                # 1. Model(s)
                for model_getter in self.model_getters:
                    import_stats, global_vars = get_import_stats_global_vars(model_getter)
                    all_import_stats.extend(import_stats)
                    all_global_vars.extend(global_vars)

                # 2. Task module
                import_stats, global_vars = get_import_stats_global_vars(self.task_module_getter)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # 3. Data module
                import_stats, global_vars = get_import_stats_global_vars(self.data_module_getter)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # 4. Runner(s)
                for runner in self._runners.values():
                    import_stats, global_vars = get_import_stats_global_vars(runner["getter"])
                    all_import_stats.extend(import_stats)
                    all_global_vars.extend(global_vars)

                # 5. Root config
                import_stats, global_vars = get_import_stats_global_vars(root_config_getter)
                all_import_stats.extend(import_stats)
                all_global_vars.extend(global_vars)

                # Process & write
                all_import_stats = set(all_import_stats)  # avoid duplicate imports
                all_global_vars = set(all_global_vars)
                f.writelines(all_import_stats)
                f.writelines(["\n", "\n"])
                if len(all_global_vars) != 0:
                    f.writelines(all_global_vars)
                    f.write("\n")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # >>>>>>>>>>>>>>>>>>>>>> Source Code >>>>>>>>>>>>>>>>>>>>>>
                for model_getter in self.model_getters:
                    f.write(inspect.getsource(model_getter) + "\n\n")
                f.write(inspect.getsource(self.task_module_getter) + "\n\n")
                f.write(inspect.getsource(self.data_module_getter) + "\n\n")

                root_config_src = inspect.getsource(root_config_getter)
                # Adapt the source code of runner(s) to solve potential name conflicts
                # when specifying multiple runners.
                for kind, runner in self._runners.items():
                    if runner["getter"]:
                        trainer_src, root_config_src = RootConfig._adapt_runner_src(
                            runner["getter"], root_config_src, kind  # noqa
                        )
                        f.write(trainer_src + "\n\n")
                f.write(root_config_src + "\n")
                # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    @rank_zero_only
    def _save_config(self, config_format: ARCHIVED_CONFIG_FORMAT, run: Run):
        # Archive config file(s)
        root_config_getter = getattr(self, "root_config_getter")
        archived_configs_dir = run.run_dir
        run_id = run.global_id

        if config_format == "single-py":
            self._archive_config_into_single(
                root_config_getter=root_config_getter,
                archived_configs_dir=archived_configs_dir,
                run_id=run_id,
            )
        elif config_format == "zip":
            self._archive_config_into_dir_or_zip(
                root_config_getter=root_config_getter,
                archived_configs_dir=archived_configs_dir,
                run_id=run_id,
                to_zip=True,
            )
        elif config_format == "dir":
            self._archive_config_into_dir_or_zip(
                root_config_getter=root_config_getter,
                archived_configs_dir=archived_configs_dir,
                run_id=run_id,
                to_zip=False,
            )
        else:
            raise ValueError(f"Unrecognized format of archived config: {config_format}")

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
