import argparse
import importlib
import os
import os.path as osp
import sys
from functools import partial
from typing import Literal

import pytorch_lightning as pl
from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from cube.c3lyr import DAOFactory, EntityFactory, EntityFSIO
from cube.config_sys import RootConfig

# Add current working directory to PYTHONPATH  # noqa
sys.path.insert(0, os.getcwd())

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Unusual Options >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# These options are unusual, do not need to be modified in most cases.

# The root directory of all output products,
# the default is the "outputs" directory in the current working directory.
OUTPUT_DIR = osp.join(os.getcwd(), "outputs")

# The format of archived configs,
#   - "SINGLE_PY": merged single .py file
#   - "ZIP": reserving original directory structure, and archive them into a .zip file
#   - "DIR": reserving original directory structure, copy them into the destination directory directly
# The default is "SINGLE_PY".
ARCHIVED_CONFIGS_FORMAT: Literal["SINGLE_PY", "ZIP", "DIR"] = "SINGLE_PY"


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


def _check_trainer(trainer: pl.Trainer, job_type: RootConfig.JOB_TYPES_T):
    if not trainer:
        raise ValueError(
            f"There is no trainer for {job_type}, please specify it in the Config instance."
        )


def _get_root_config_instance(config_file_path, return_getter=False):
    # Remove the part of root directory in an absolute path.
    relative_file_path = config_file_path.replace(osp.split(__file__)[0], "")
    import_path = relative_file_path.replace("\\", ".").replace("/", ".")

    # Remove the opening '.', or importlib will seem it as a relative import and the `package` argument is needed.
    if import_path.startswith("."):
        import_path = import_path[1:]
    if import_path.endswith(".py"):
        import_path = import_path[:-3]

    try:
        root_config_getter = getattr(
            importlib.import_module(import_path), RootConfig.ROOT_CONFIG_GETTER_NAME
        )
        config_instance: RootConfig = root_config_getter()
    except AttributeError:
        raise AttributeError(
            f"Expected a function called `{RootConfig.ROOT_CONFIG_GETTER_NAME}` in your root config "
            f"file, like `def {RootConfig.ROOT_CONFIG_GETTER_NAME}(): ...`"
        )
    return (config_instance, root_config_getter) if return_getter else config_instance


def init(args: argparse.Namespace):
    # >>>>>>>>>>>>>>>>>>> Preprocess `logger` argument >>>>>>>>>>>>>>>>>>>
    if isinstance(args.logger, str):
        logger = args.logger.lower()
        if logger == "true":
            logger = "csv"  # default to "csv" when given "true"
        elif logger not in ("false", "csv"):
            logger = ["csv", logger]  # add "csv" automatically
        # For "false"/"csv", do nothing.

    # A list
    else:
        # Deduplicate
        logger = list({lg.lower() for lg in args.logger})

        # Ignore "false"/"true" when give multiple
        if "false" in logger:
            logger.remove("false")
        if "true" in logger:
            logger.remove("true")

        # Make "csv" always be the first, as some callbacks (e.g. `ModelCheckpoint`) depends on
        # the first logger's path setting.
        if "csv" not in logger:
            logger.insert(0, "csv")
        else:
            logger.remove("csv")
            logger.insert(0, "csv")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    proj = EntityFactory.get_proj_instance(
        name=args.proj_name, desc=args.proj_desc, output_dir=OUTPUT_DIR, logger=logger
    )
    DAOFactory.get_proj_dao().insert_entry(proj)
    exp = EntityFactory.get_exp_instance(
        name=args.exp_name,
        desc=args.exp_desc,
        output_dir=OUTPUT_DIR,
        proj_id=proj.global_id,
    )
    DAOFactory.get_exp_dao().insert_entry(exp)


def add_exp(args: argparse.Namespace):
    exp = EntityFactory.get_exp_instance(
        name=args.exp_name,
        desc=args.exp_desc,
        output_dir=OUTPUT_DIR,
        proj_id=args.proj_id,
    )
    DAOFactory.get_exp_dao().insert_entry(exp)


def ls(args: argparse.Namespace):
    def _draw_table(rich_table, items, prompt_on_empty):
        if len(items) > 0:
            for column_name in items[0].keys():
                rich_table.add_column(column_name, overflow="fold")
            for item in items:
                rich_table.add_row(*list(item.values()))
        else:
            print(prompt_on_empty)

    def _draw_nested_table(
        rich_table,
        items,
        inner_table_key,
        prompt_on_empty,
        width_ratios=(8, 12, 20, 10, 50),
    ):
        if len(items) > 0:
            # Add header to the outer table
            for idx, outer_column_name in enumerate(items[0].keys()):
                rich_table.add_column(
                    outer_column_name, overflow="fold", ratio=width_ratios[idx]
                )

            for item in items:
                inner_table = Table(show_header=True, header_style="bold green")
                if len(item[inner_table_key]) > 0:
                    # Add header to the inner table
                    for idx, inner_column_name in enumerate(
                        item[inner_table_key][0].keys()
                    ):
                        inner_table.add_column(
                            inner_column_name, overflow="fold", ratio=width_ratios[idx]
                        )
                    # Add rows to the inner table
                    for exp in item[inner_table_key]:
                        inner_table.add_row(*list(exp.values()))

                # Add rows to the outer table
                rich_table.add_row(*list(item.values())[:-1], Columns([inner_table]))
        else:
            print(prompt_on_empty)

    console = Console()
    get_table = partial(
        Table, show_header=True, header_style="bold blue", width=console.width
    )

    # Print all projects and exps
    if args.projs_exps:
        proj_dao = DAOFactory.get_proj_dao()
        table = get_table(title=f'All Projects and Exps in "{OUTPUT_DIR}"')
        projects_exps = proj_dao.get_all_projects_exps(OUTPUT_DIR)
        _draw_nested_table(
            table,
            projects_exps,
            inner_table_key="Exps",
            prompt_on_empty=f"There is no proj/exp in {OUTPUT_DIR}.",
        )

    # Print all projects
    elif args.projs:
        proj_dao = DAOFactory.get_proj_dao()
        table = get_table(title=f'All Projects in "{OUTPUT_DIR}"')
        _draw_table(
            table,
            proj_dao.get_projects(OUTPUT_DIR),
            f'There is no project in "{OUTPUT_DIR}".',
        )

    elif args.exps_runs_of:
        exp_dao = DAOFactory.get_exp_dao()
        table = get_table(
            title=f'All Exps and Runs of Proj "{args.exps_runs_of}" in "{OUTPUT_DIR}"'
        )
        exps_runs = exp_dao.get_all_exps_runs(OUTPUT_DIR, proj_id=args.exps_runs_of)
        _draw_nested_table(
            table,
            exps_runs,
            inner_table_key="Runs",
            prompt_on_empty=f'There is no exp/run of proj "{args.exps_runs_of}" in {OUTPUT_DIR}.',
        )

    # Print all exps of the specified proj
    elif args.exps_of:
        exp_dao = DAOFactory.get_exp_dao()
        table = get_table(title=f'All Exps of Proj "{args.exps_of}" in "{OUTPUT_DIR}"')
        _draw_table(
            table,
            exp_dao.get_exps_of(OUTPUT_DIR, proj_id=args.exps_of),
            f'There is no exp of proj "{args.exps_of}" in "{OUTPUT_DIR}".',
        )

    # Print all runs of the specified exp of the specified proj
    else:
        run_dao = DAOFactory.get_run_dao()
        table = get_table(
            title=f'All Runs of exp "{args.runs_of[1]}" of '
            f'proj "{args.runs_of[0]}" in "{OUTPUT_DIR}"'
        )
        _draw_table(
            table,
            run_dao.get_runs_of(
                OUTPUT_DIR, proj_id=args.runs_of[0], exp_id=args.runs_of[1]
            ),
            f'There is no run of exp "{args.runs_of[1]}" of '
            f'proj "{args.runs_of[0]}" in "{OUTPUT_DIR}".',
        )

    console.print(table)


def _exec(args: argparse.Namespace, job_type: RootConfig.JOB_TYPES_T):
    root_config_instance, root_config_getter = _get_root_config_instance(
        args.config_file, return_getter=True
    )

    pl.seed_everything(root_config_instance.global_seed)

    # Set up the task wrapper and the data wrapper.
    root_config_instance.setup_wrappers()

    run_dao = DAOFactory.get_run_dao()
    # When resuming fit, the run should resume from the original.
    if job_type == "resume-fit":
        proj_id, exp_id, run_id = run_dao.parse_ids_from_ckpt_path(args.resume_from)
        run = run_dao.get_run_from_id(
            OUTPUT_DIR, proj_id=proj_id, exp_id=exp_id, run_id=run_id
        )
        run.is_resuming = True
        EntityFSIO.process_metrics_csv(run.run_dir)

    # For any other run, a new one should be created.
    else:
        run = EntityFactory.get_run_instance(
            name=args.name,
            desc=args.desc,
            output_dir=OUTPUT_DIR,
            proj_id=args.proj_id,
            exp_id=args.exp_id,
            job_type=job_type,
        )
        run_dao.insert_entry(run)
        run.is_resuming = False

        # >>>>>>>>>>>>>>>>>>>>> Archive the Config Files >>>>>>>>>>>>>>>>>>>>>
        run_id = run.global_id
        archived_configs_dir = run.run_dir
        if ARCHIVED_CONFIGS_FORMAT == "SINGLE_PY":
            root_config_instance.archive_config_into_single(
                root_config_getter,
                archived_configs_dir=archived_configs_dir,
                run_id=run_id,
            )
        elif ARCHIVED_CONFIGS_FORMAT == "ZIP":
            root_config_instance.archive_config_into_dir_or_zip(
                root_config_getter,
                archived_configs_dir=archived_configs_dir,
                run_id=run_id,
            )
        elif ARCHIVED_CONFIGS_FORMAT == "DIR":
            root_config_instance.archive_config_into_dir_or_zip(
                root_config_getter,
                archived_configs_dir=archived_configs_dir,
                run_id=run_id,
                to_zip=False,
            )
        else:
            raise ValueError(
                f"Unrecognized ARCHIVED_CONFIGS_FORMAT: {ARCHIVED_CONFIGS_FORMAT}"
            )
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Set preset environment variables
    os.environ["CUBE_RUN_DIR"] = run.run_dir

    # Set up the trainer(s) for the current run.
    if args.off_log:
        print("Logging has been off.")
        root_config_instance.setup_trainer(logger_arg="false", run=run)
    else:
        root_config_instance.setup_trainer(
            logger_arg=run.belonging_exp.belonging_proj.logger, run=run
        )

    if job_type == "fit":
        _check_trainer(root_config_instance.fit_trainer, job_type)
        root_config_instance.fit_trainer.fit(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper,
        )
        # Merge multiple metrics_csv (if any) into "merged_metrics.csv" after fit exits
        # (finishes / KeyboardInterrupt).
        EntityFSIO.merge_metrics_csv(run.run_dir)

    elif job_type == "resume-fit":
        _check_trainer(root_config_instance.fit_trainer, job_type)
        root_config_instance.fit_trainer.fit(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper,
            ckpt_path=args.resume_from,
        )
        EntityFSIO.merge_metrics_csv(run.run_dir)

    elif job_type == "validate":
        _check_trainer(root_config_instance.validate_trainer, job_type)
        if args.loaded_ckpt is not None:
            # Load the specified checkpoint
            loaded_task_wrapper = (
                root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                    args.loaded_ckpt,
                    **root_config_instance.task_wrapper.get_init_args(),
                )
            )
            root_config_instance.task_wrapper = (
                loaded_task_wrapper  # update the task wrapper
            )

        run_dao.set_extra_data(
            run, loaded_ckpt=args.loaded_ckpt
        )  # Save the loaded ckpt info
        root_config_instance.validate_trainer.validate(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper,
        )

    elif job_type == "test":
        _check_trainer(root_config_instance.test_trainer, job_type)
        # Same as "validate"
        if args.loaded_ckpt is not None:
            loaded_task_wrapper = (
                root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                    args.loaded_ckpt,
                    **root_config_instance.task_wrapper.get_init_args(),
                )
            )
            root_config_instance.task_wrapper = loaded_task_wrapper

        run_dao.set_extra_data(run, loaded_ckpt=args.loaded_ckpt)
        root_config_instance.test_trainer.test(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper,
        )

    elif job_type == "predict":
        _check_trainer(root_config_instance.predict_trainer, job_type)
        # Same as "validate"
        if args.loaded_ckpt is not None:
            loaded_task_wrapper = (
                root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                    args.loaded_ckpt,
                    **root_config_instance.task_wrapper.get_init_args(),
                )
            )
            root_config_instance.task_wrapper = loaded_task_wrapper

        run_dao.set_extra_data(run, loaded_ckpt=args.loaded_ckpt)
        predictions = root_config_instance.predict_trainer.predict(
            model=root_config_instance.task_wrapper,
            datamodule=root_config_instance.data_wrapper,
        )
        root_config_instance.task_wrapper.save_predictions(
            predictions, EntityFSIO.mkdir_for_predictions(run.run_dir)
        )

    else:
        raise ValueError(f"Unrecognized job type: {job_type}!")

    # Remove empty "hparams.yaml" generated by PyTorch-Lightning
    EntityFSIO.remove_empty_hparams_yaml(run.run_dir)

    # Delete preset environment variables
    del os.environ["CUBE_RUN_DIR"]


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # >>>>>>>>>>>>>>>>>>>>>>> Subcommand 1: init >>>>>>>>>>>>>>>>>>>>>>>
    parser_init = subparsers.add_parser("init", help="Create a new proj and a new exp.")
    parser_init.add_argument(
        "-pn",
        "--proj-name",
        "--proj_name",
        type=str,
        required=True,
        help="Name of the new proj.",
    )
    parser_init.add_argument(
        "-pd",
        "--proj-desc",
        "--proj_desc",
        type=str,
        required=True,
        help="Description of the new proj.",
    )
    parser_init.add_argument(
        "-en",
        "--exp-name",
        "--exp_name",
        type=str,
        required=True,
        help="Name of the new exp.",
    )
    parser_init.add_argument(
        "-ed",
        "--exp-desc",
        "--exp_desc",
        type=str,
        required=True,
        help="Description of the new exp.",
    )
    parser_init.add_argument(
        "-l",
        "--logger",
        type=str,
        default="CSV",
        nargs="*",
        help=f"Choose one or multiple from {RootConfig.LOGGERS} "
        f'and combine them arbitrarily e.g. "tensorboard wandb". '
        f"Or it can be True/False, meaning using the default CSV and "
        f"disable logging respectively. Note that CSV will be always "
        f"added automatically when it is not False.",
    )
    parser_init.set_defaults(func=init)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>> Subcommand 2: add-exp >>>>>>>>>>>>>>>>>>>>>>>
    parser_exp = subparsers.add_parser(
        "add-exp", help="Create a new exp within specified proj."
    )
    parser_exp.add_argument(
        "-p",
        "--proj-id",
        "--proj_id",
        type=str,
        required=True,
        help="ID of the proj that the new exp belongs to.",
    )
    parser_exp.add_argument(
        "-en",
        "--exp-name",
        "--exp_name",
        type=str,
        required=True,
        help="Name of the new exp.",
    )
    parser_exp.add_argument(
        "-ed",
        "--exp-desc",
        "--exp_desc",
        type=str,
        required=True,
        help="Description of the new exp.",
    )
    parser_exp.set_defaults(func=add_exp)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>> Subcommand 3: ls >>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_ls = subparsers.add_parser(
        "ls", help="Display info about proj, exp and runs in tables."
    )
    # These params are exclusive to each other, and one of them is required.
    param_group_ls = parser_ls.add_mutually_exclusive_group(required=True)
    param_group_ls.add_argument(
        "-pe",
        "--projs-exps",
        "--projs_exps",
        action="store_true",
        help="Display all projs and exps.",
    )
    param_group_ls.add_argument(
        "-p", "--projs", action="store_true", help="Display all projs."
    )
    param_group_ls.add_argument(
        "-er",
        "--exps-runs-of",
        "--exps_runs_of",
        type=str,
        help="Display all exps and runs of the proj specified by ID.",
    )
    param_group_ls.add_argument(
        "-e",
        "--exps-of",
        "--exps_of",
        type=str,
        help="Display all exps of the proj specified by ID.",
    )
    param_group_ls.add_argument(
        "-r",
        "--runs-of",
        "--runs_of",
        type=str,
        nargs=2,
        help="Display all runs of the exp of the proj specified by IDs.",
    )
    parser_ls.set_defaults(func=ls)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>> Subcommands: fit, resume-fit, validate, test, predict >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Create a parent subparser containing common arguments.
    # Ref: https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers
    exec_parent_parser = argparse.ArgumentParser(add_help=False)
    exec_parent_parser.add_argument(
        "-c",
        "--config-file",
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    exec_parent_parser.add_argument(
        "-p",
        "--proj-id",
        "--proj_id",
        type=str,
        required=True,
        help="ID of the proj ID that the new run belongs to.",
    )
    exec_parent_parser.add_argument(
        "-e",
        "--exp-id",
        "--exp_id",
        type=str,
        required=True,
        help="ID of the exp that the new run belongs to.",
    )
    exec_parent_parser.add_argument(
        "-n", "--name", type=str, required=True, help="Name of the new run."
    )
    exec_parent_parser.add_argument(
        "-d", "--desc", type=str, required=True, help="Description of the new run."
    )
    exec_parent_parser.add_argument(
        "-o",
        "--off-log",
        "-off_log",
        action="store_true",
        help="Turn off all logging during the current execution.",
    )

    # >>>>>>>>>>>>>> Subcommand 4: fit >>>>>>>>>>>>>>>
    parser_fit = subparsers.add_parser(
        "fit",
        parents=[exec_parent_parser],
        help="Execute a fit run on the dataset's fit split.",
    )
    parser_fit.set_defaults(func=partial(_exec, job_type="fit"))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>> Subcommand 5: resume-fit >>>>>>>>>>>
    parser_resume_fit = subparsers.add_parser(
        "resume-fit", help="Resume an interrupted fit run."
    )
    parser_resume_fit.add_argument(
        "-c",
        "--config-file",
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser_resume_fit.add_argument(
        "-r",
        "--resume-from",
        "--resume_from",
        type=str,
        required=True,
        help="File path to the checkpoint where resumes.",
    )
    parser_resume_fit.set_defaults(func=partial(_exec, job_type="resume-fit"))
    parser_resume_fit.add_argument(
        "-o",
        "--off-log",
        "-off_log",
        action="store_true",
        help="Turn off all logging during the current execution.",
    )
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>> Subcommands: validate, test and predict >>>>>>>>>>>>>>>>>>>>>>
    # Common parent parser.
    validate_test_predict_parent_parser = argparse.ArgumentParser(add_help=False)
    validate_test_predict_parent_parser.add_argument(
        "-lc",
        "--loaded-ckpt",
        "--loaded_ckpt",
        type=str,
        default=None,
        help="File path to the loaded model checkpoint. Default is None, "
        "which means you are going to conduct validate/test/predict "
        "using the initialized model without loading any weights).",
    )
    # >>>>>>>>>>> Subcommand 6: validate >>>>>>>>>>>
    parser_validate = subparsers.add_parser(
        "validate",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a validate run on the dataset's validate split.",
    )
    parser_validate.set_defaults(func=partial(_exec, job_type="validate"))

    # >>>>>>>>>>> Subcommand 7: test >>>>>>>>>>>
    parser_test = subparsers.add_parser(
        "test",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a test run on the dataset's test split.",
    )
    parser_test.set_defaults(func=partial(_exec, job_type="test"))

    # >>>>>>>>>>> Subcommand 8: predict >>>>>>>>>>>
    parser_predict = subparsers.add_parser(
        "predict",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a predict run.",
    )
    parser_predict.set_defaults(func=partial(_exec, job_type="predict"))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # Parse args and invoke the corresponding function
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
