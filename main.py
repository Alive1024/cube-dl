import os.path as osp
import argparse
import importlib
from typing import Literal
from functools import partial

import pytorch_lightning as pl
from rich.console import Console
from rich.table import Table
from rich.columns import Columns

from root_config import RootConfig
from entities import (Project, Experiment, Run, get_all_projects_exps, get_projects, get_exps_of, get_runs_of)

# ========================== Unusual Options ==========================
# These options are unusual, do not need to be modified in most cases.

# The root directory of all output products,
# the default is the "outputs" directory in the entire project root directory.
OUTPUT_DIR = osp.join(osp.dirname(osp.splitext(__file__)[0]), "outputs")

# The global seed of random numbers.
GLOBAL_SEED = 42

# The format of archived configs,
#   - "SINGLE_PY":
#   - "ZIP":
#   - "DIR":
# The default is "SINGLE_PY"
ARCHIVED_CONFIGS_FORMAT: Literal["SINGLE_PY", "ZIP", "DIR"] = "SINGLE_PY"
# =======================================================================


def __check_trainer(trainer: pl.Trainer, job_type: RootConfig.JOB_TYPES_T):
    if not trainer:
        raise ValueError(f"There is no trainer for {job_type}, please specify it in the Config instance.")


def __get_root_config_instance(config_file_path, return_getter=False):
    # Remove the part of root directory in an absolute path.
    relative_file_path = config_file_path.replace(osp.split(__file__)[0], '')
    import_path = relative_file_path.replace('\\', '.').replace('/', '.')

    # Remove the opening '.', or importlib will seem it as a relative import and the `package` argument is needed.
    if import_path.startswith('.'):
        import_path = import_path[1:]
    if import_path.endswith(".py"):
        import_path = import_path[:-3]

    try:
        root_config_getter = getattr(importlib.import_module(import_path), RootConfig.CONFIG_GETTER_NAME)
        config_instance: RootConfig = root_config_getter()
    except AttributeError:
        raise AttributeError(f"Expected a function called `{RootConfig.CONFIG_GETTER_NAME}` in your root config file, "
                             f"like `def {RootConfig.CONFIG_GETTER_NAME}(): ...`")
    return (config_instance, root_config_getter) if return_getter else config_instance


def _init(args: argparse.Namespace):
    proj = Project(name=args.proj_name, desc=args.proj_desc, output_dir=OUTPUT_DIR)
    Experiment(name=args.exp_name, desc=args.exp_desc, proj_id=proj.global_id, output_dir=OUTPUT_DIR)


def _add_exp(args: argparse.Namespace):
    Experiment(name=args.exp_name, desc=args.exp_desc, proj_id=args.proj_id, output_dir=OUTPUT_DIR)


def _ls(args: argparse.Namespace):
    def _draw_table(rich_table, items, prompt_on_empty):
        if len(items) > 0:
            for column_name in items[0].keys():
                rich_table.add_column(column_name, overflow="fold")
            for item in items:
                rich_table.add_row(*list(item.values()))
        else:
            print(prompt_on_empty)

    console = Console()
    get_table = partial(Table, show_header=True, header_style="bold blue", width=console.width)

    # Print all projects and exps
    if args.all:
        table = get_table(title=f"All Projects and Exps in \"{OUTPUT_DIR}\"")
        projects_exps = get_all_projects_exps(OUTPUT_DIR)
        ratios = (8, 12, 20, 10, 50)
        if len(projects_exps) > 0:
            # Add header to the outer table
            for idx, outer_column_name in enumerate(projects_exps[0].keys()):
                table.add_column(outer_column_name, overflow="fold", ratio=ratios[idx])

            for proj in projects_exps:
                inner_table = Table(show_header=True, header_style="bold green")
                if len(proj["Exps"]) > 0:
                    # Add header to the inner table
                    for idx, inner_column_name in enumerate(proj["Exps"][0].keys()):
                        inner_table.add_column(inner_column_name, overflow="fold", ratio=ratios[idx])
                    # Add rows to the inner table
                    for exp in proj["Exps"]:
                        inner_table.add_row(*list(exp.values()))

                # Add rows to the outer table
                table.add_row(*list(proj.values())[:-1], Columns([inner_table]))
        else:
            print(f"There is no proj/exp in {OUTPUT_DIR}.")

    # Print all projects
    elif args.proj:
        table = get_table(title=f"All Projects in \"{OUTPUT_DIR}\"")
        _draw_table(table, get_projects(OUTPUT_DIR),
                    f"There is no project in \"{OUTPUT_DIR}\".")
    # Print all exps of the specified proj
    elif args.exp_of:
        table = get_table(title=f"All Exps of Proj \"{args.exp_of}\" in \"{OUTPUT_DIR}\"")
        _draw_table(table, get_exps_of(OUTPUT_DIR, args.exp_of),
                    f"There is no exp of proj \"{args.exp_of}\" in \"{OUTPUT_DIR}\".")

    # Print all runs of the specified exp of the specified proj
    else:
        table = get_table(title=f"All Runs of exp \"{args.run_of[1]}\" of "
                                f"proj \"{args.run_of[0]}\" in \"{OUTPUT_DIR}\"")
        _draw_table(table, get_runs_of(OUTPUT_DIR, proj_id=args.run_of[0], exp_id=args.run_of[1]),
                    f"There is no run of exp \"{args.run_of[1]}\" of "
                    f"proj \"{args.run_of[0]}\" in \"{OUTPUT_DIR}\".")

    console.print(table)


def __exec(args: argparse.Namespace, job_type: RootConfig.JOB_TYPES_T):
    pl.seed_everything(GLOBAL_SEED)
    root_config_instance, root_config_getter = __get_root_config_instance(args.config_file, return_getter=True)

    # ============================ Executing the Run ============================
    # Set up the task wrapper and the data wrapper.
    root_config_instance.setup_wrappers()

    run = Run(name=args.name, desc=args.desc, proj_id=args.proj_id, exp_id=args.exp_id,
              job_type=job_type, output_dir=OUTPUT_DIR, resume_from=args.resumes_from if job_type == "fit" else None)

    # ======= Archive the Config Files =======
    # When resuming fit, there will be no new run.
    if not (job_type == "fit" and args.resumes_from):
        run_id = run.global_id
        archived_configs_dir = Experiment.get_archived_configs_dir(proj_id=args.proj_id, exp_id=args.exp_id,
                                                                   output_dir=OUTPUT_DIR)
        if ARCHIVED_CONFIGS_FORMAT == "SINGLE_PY":
            root_config_instance.archive_config_into_single(root_config_getter,
                                                            archived_configs_dir=archived_configs_dir,
                                                            run_id=run_id)
        elif ARCHIVED_CONFIGS_FORMAT == "ZIP":
            root_config_instance.archive_config_into_dir_or_zip(root_config_getter,
                                                                archived_configs_dir=archived_configs_dir,
                                                                run_id=run_id)
        elif ARCHIVED_CONFIGS_FORMAT == "DIR":
            root_config_instance.archive_config_into_dir_or_zip(root_config_getter,
                                                                archived_configs_dir=archived_configs_dir,
                                                                run_id=run_id,
                                                                to_zip=False)
        else:
            raise ValueError(f"Unrecognized ARCHIVED_CONFIGS_FORMAT: {ARCHIVED_CONFIGS_FORMAT}")

    # Set up the trainer(s) for the current run.
    root_config_instance.setup_trainer(logger_arg=args.logger, run=run)

    if job_type == "fit":
        __check_trainer(root_config_instance.fit_trainer, job_type)
        root_config_instance.fit_trainer.fit(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper,
            ckpt_path=args.resumes_from
        )
        # Merge multiple metrics_csv (if any) into "merged_metrics.csv" after fit exits
        # (finishes / KeyboardInterrupt).
        Run.merge_metrics_csv(run.run_dir)

    elif job_type == "validate":
        __check_trainer(root_config_instance.validate_trainer, job_type)
        if args.loaded_ckpt != "none":
            # Load the specified checkpoint
            loaded_task_wrapper = root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                args.loaded_ckpt,
                **root_config_instance.task_wrapper.get_init_args()
            )
            root_config_instance.task_wrapper = loaded_task_wrapper  # update the task wrapper

        run.set_extra_record_data(loaded_ckpt=args.loaded_ckpt)
        root_config_instance.validate_trainer.validate(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper
        )

    elif job_type == "test":
        __check_trainer(root_config_instance.test_trainer, job_type)
        # Same as "validate"
        if args.loaded_ckpt != "none":
            loaded_task_wrapper = root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                args.loaded_ckpt,
                **root_config_instance.task_wrapper.get_init_args()
            )
            root_config_instance.task_wrapper = loaded_task_wrapper

        run.set_extra_record_data(loaded_ckpt=args.loaded_ckpt)
        root_config_instance.test_trainer.test(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper
        )

    elif job_type == "predict":
        __check_trainer(root_config_instance.predict_trainer, job_type)
        # Same as "validate"
        if args.loaded_ckpt != "none":
            loaded_task_wrapper = root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                args.loaded_ckpt,
                **root_config_instance.task_wrapper.get_init_args()
            )
            root_config_instance.task_wrapper = loaded_task_wrapper

        run.set_extra_record_data(loaded_ckpt=args.loaded_ckpt)
        predictions = root_config_instance.predict_trainer.predict(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper
        )
        root_config_instance.task_wrapper.save_predictions(predictions, Run.mkdir_for_predictions(run.run_dir))

    elif job_type == "tune":
        # TODO
        raise NotImplementedError

    else:
        raise ValueError(f"Unrecognized job type: {job_type}!")

    # Remove empty "hparams.yaml" generated by PyTorch-Lightning
    Run.remove_empty_hparams_yaml(run.run_dir)
    # ============================================================================


def _fit(args: argparse.Namespace):
    __exec(args, "fit")


def _validate(args: argparse.Namespace):
    __exec(args, "validate")


def _test(args: argparse.Namespace):
    __exec(args, "test")


def _predict(args: argparse.Namespace):
    __exec(args, "predict")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # ======================= Subcommand: init =======================
    parser_init = subparsers.add_parser("init", help="")
    parser_init.add_argument("-pn", "--proj-name", "--proj_name", type=str, required=True,
                             help="")
    parser_init.add_argument("-pd", "--proj-desc", "--proj_desc", type=str, required=True,
                             help="")
    parser_init.add_argument("-en", "--exp-name", "--exp_name", type=str, required=True,
                             help="")
    parser_init.add_argument("-ed", "--exp-desc", "--exp_desc", type=str, required=True,
                             help="")
    parser_init.set_defaults(func=_init)

    # ======================= Subcommand: add-exp =======================
    parser_exp = subparsers.add_parser("add-exp", help="Create a new exp within specified proj.")
    parser_exp.add_argument("-p", "--proj-id", "--proj_id", type=str, required=True,
                            help="")
    parser_exp.add_argument("-en", "--exp-name", "--exp_name", type=str, required=True,
                            help="The name of the experiment, it will be appended to the prefix: exp_{exp_id}.")
    parser_exp.add_argument("-ed", "--exp-desc", "--exp_desc", type=str, required=True,
                            help="A description about the created exp.")
    parser_exp.set_defaults(func=_add_exp)

    # ======================= Subcommand: ls =======================
    parser_ls = subparsers.add_parser("ls", help="")
    # These params are exclusive to each other, and one of them is required.
    param_group_ls = parser_ls.add_mutually_exclusive_group(required=True)
    param_group_ls.add_argument("-a", "--all", action="store_true",
                                help="")
    param_group_ls.add_argument("-p", "--proj", action="store_true",
                                help="")
    param_group_ls.add_argument("-e", "--exp-of", "--exp_of", type=str, default=None,
                                help="")
    param_group_ls.add_argument("-r", "--run-of", "--run_of", type=str, default=None, nargs=2,
                                help="")
    parser_ls.set_defaults(func=_ls)

    # ======================= Subcommands for exec =======================
    # Create a parent subparser containing common arguments.
    # Ref: https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers
    exec_parent_parser = argparse.ArgumentParser(add_help=False)
    exec_parent_parser.add_argument("-c", "--config-file", "--config_file", type=str, required=True,
                                    help="Path to the config file")
    exec_parent_parser.add_argument("-p", "--proj-id", "--proj_id", type=str, required=True,
                                    help="The project id of which the current run belongs to.")
    exec_parent_parser.add_argument("-e", "--exp-id", "--exp_id", type=str, required=True,
                                    help="The experiment id of which the current run belongs to.")
    exec_parent_parser.add_argument("-n", "--name", type=str, required=True,
                                    help="The name of the current run.")
    exec_parent_parser.add_argument("-d", "--desc", type=str, required=True,
                                    help="A description about the current run.")
    exec_parent_parser.add_argument("-l", "--logger", type=str, default="CSV",
                                    help=f"Choose from {RootConfig.LOGGERS} "
                                         f"and combine arbitrarily, seperated by comma. e.g. \"csv,wandb\" "
                                         f"Or it can be True/False, meaning using the default CSV and "
                                         f"disable logging respectively.")

    # ========== Subcommand: fit ==========
    parser_fit = subparsers.add_parser("fit", parents=[exec_parent_parser],
                                       help="")
    parser_fit.add_argument("-r", "--resumes-from", "--resumes_from", type=str, default=None,
                            help="")
    parser_fit.set_defaults(func=_fit)

    # ========== Subcommands: validate, test and predict ==========
    # Common parent parser.
    validate_test_predict_parent_parser = argparse.ArgumentParser(add_help=False)
    validate_test_predict_parent_parser.add_argument("-lc", "--loaded-ckpt", "--loaded_ckpt", type=str, required=True,
                                                     help="This must be explicitly provided It can be a file path, "
                                                          "or \"none\" (this means you are going to conduct "
                                                          "validate/test/predict using the initialized model "
                                                          "without loading any weights).")
    # ========== Subcommand: validate ==========
    parser_validate = subparsers.add_parser("validate", parents=[exec_parent_parser,
                                                                 validate_test_predict_parent_parser],
                                            help="")
    parser_validate.set_defaults(func=_validate)

    # ========== Subcommand: test ==========
    parser_test = subparsers.add_parser("test", parents=[exec_parent_parser, validate_test_predict_parent_parser],
                                        help="")
    parser_test.set_defaults(func=_test)

    # ========== Subcommand: predict ==========
    parser_predict = subparsers.add_parser("predict", parents=[exec_parent_parser, validate_test_predict_parent_parser],
                                           help="")
    parser_predict.set_defaults(func=_predict)

    # Parse args and invoke the corresponding function
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
