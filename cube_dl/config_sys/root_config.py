import ast
import importlib
import inspect
import os
import os.path as osp
import shutil
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Literal

from torch.nn import Module

from cube_dl.c3lyr import Run
from cube_dl.callback import CubeCallback, CubeCallbackList
from cube_dl.core import CubeDataModule, CubeRunner, CubeTaskModule

ARCHIVED_CONFIG_FORMAT = Literal["single-py", "zip", "dir"] | bool
RUNNER_GETTER_T = Callable[[], CubeRunner]
RUNNER_TYPES_T = Literal["default", "fit", "validate", "test", "predict", "tune"]


def get_root_config_instance(config_file_path, return_getter=False):
    # Remove the part of root directory in an absolute path.
    relative_file_path = config_file_path.replace(osp.split(__file__)[0], "")
    import_path = relative_file_path.replace("\\", ".").replace("/", ".")

    # Remove the opening '.', or importlib will seem it as a relative import and the `package` argument is needed.
    if import_path.startswith("."):
        import_path = import_path[1:]
    if import_path.endswith(".py"):
        import_path = import_path[:-3]

    try:
        root_config_getter = getattr(importlib.import_module(import_path), RootConfig.ROOT_CONFIG_GETTER_NAME)
        root_config: RootConfig = root_config_getter()
        # Attach the `root_config_getter` to the `root_config` instance
        setattr(root_config, "root_config_getter", root_config_getter)
    except AttributeError:
        raise AttributeError(
            f"Expect a function called `{RootConfig.ROOT_CONFIG_GETTER_NAME}` as the root config getter "
            f"in the config file, like `def {RootConfig.ROOT_CONFIG_GETTER_NAME}(): ...`"
        )
    return (root_config, root_config_getter) if return_getter else root_config


class RootConfig:
    # The uniform name of the root config getter
    ROOT_CONFIG_GETTER_NAME = "get_root_config"

    def __init__(
        self,
        *,  # Compulsory keyword arguments, for better readability in config files.
        model_getters: Callable[[], Module] | Iterable[Callable[[], Module]],
        task_module_getter: Callable[[], CubeTaskModule],
        data_module_getter: Callable[[], CubeDataModule],
        default_runner_getter: RUNNER_GETTER_T | None = None,
        fit_runner_getter: RUNNER_GETTER_T | None = None,
        validate_runner_getter: RUNNER_GETTER_T | None = None,
        test_runner_getter: RUNNER_GETTER_T | None = None,
        predict_runner_getter: RUNNER_GETTER_T | None = None,
        seed_func: Callable[[int | None], Any],
        global_seed: int | None = 42,
        archive_config: ARCHIVED_CONFIG_FORMAT = "single-py",
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

        seed_func(global_seed)
        self.global_seed = global_seed
        self.archive_config = archive_config
        self.callbacks: CubeCallbackList = CubeCallbackList(callbacks)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Methods for Setting up >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def setup_task_data_modules(self):
        """Instantiate the task/data modules."""
        self.task_module = self.task_module_getter()
        self.data_module = self.data_module_getter()

    def setup_runners(self, run: Run):  # noqa: C901
        """Set up the runner(s) using the getters."""
        job_type = run.job_type

        if job_type == "fit":
            if self.fit_runner_getter is None:
                self.fit_runner_getter = self.default_runner_getter
            if self.fit_runner_getter is None:
                raise ValueError(f"There is no runner for {job_type}, please specify it in the root config.")
            self.fit_runner = self.fit_runner_getter()
        elif job_type == "validate":
            if self.validate_runner_getter is None:
                self.validate_runner_getter = self.default_runner_getter
            if self.validate_runner_getter is None:
                raise ValueError(f"There is no runner for {job_type}, please specify it in the root config.")
            self.validate_runner = self.validate_runner_getter()
        elif job_type == "test":
            if self.test_runner_getter is None:
                self.test_runner_getter = self.default_runner_getter
            if self.test_runner_getter is None:
                raise ValueError(f"There is no runner for {job_type}, please specify it in the root config.")
            self.test_runner = self.test_runner_getter()
        elif job_type == "predict":
            if self.predict_runner_getter is None:
                self.predict_runner_getter = self.default_runner_getter
            if self.predict_runner_getter is None:
                raise ValueError(f"There is no runner for {job_type}, please specify it in the root config.")
            self.predict_runner = self.predict_runner_getter()

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

    def _save_config(self, config_format: ARCHIVED_CONFIG_FORMAT, run: Run):
        # No saving config
        if not config_format:
            return

        # Archive config file(s)
        root_config_getter = getattr(self, "root_config_getter")
        archived_configs_dir = run.run_dir
        run_id = run.global_id

        if config_format is True or config_format == "single-py":
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
