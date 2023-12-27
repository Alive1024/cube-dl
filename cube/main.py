import argparse
import os
import os.path as osp
import shutil
import sys
import warnings
from functools import partial
from typing import Any, Literal

from rich.columns import Columns
from rich.console import Console
from rich.table import Table

from cube.c3lyr import DAOFactory, EntityFactory, dump_run, load_run, remove_run
from cube.config_sys import get_root_config_instance
from cube.core import CUBE_CONTEXT
from cube.utils import parse_cube_configs

# Add current working directory to PYTHONPATH  # noqa
sys.path.insert(0, os.getcwd())

CUBE_CONFIGS: dict[str, Any] | None = None
JOB_TYPES_T = Literal["fit", "resume-fit", "validate", "test", "predict", "tune"]


def _parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # >>>>>>>>>>>>>>>>>>>>>>> Subcommand: init >>>>>>>>>>>>>>>>>>>>>>>
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
        required=False,
        default="",
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
        required=False,
        default="",
        help="Description of the new exp.",
    )
    parser_init.set_defaults(func=init)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>> Subcommand: add-exp >>>>>>>>>>>>>>>>>>>>>>>
    parser_exp = subparsers.add_parser("add-exp", help="Create a new exp within specified proj.")
    parser_exp.add_argument(
        "-p",
        "--proj-id",
        "--proj_id",
        type=str,
        required=True,
        help="ID of the proj that the new exp belongs to.",
    )
    parser_exp.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name of the new exp.",
    )
    parser_exp.add_argument(
        "-d",
        "--desc",
        type=str,
        required=False,
        default="",
        help="Description of the new exp.",
    )
    parser_exp.set_defaults(func=add_exp)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>> Subcommand: ls >>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_ls = subparsers.add_parser("ls", help="Display info about proj, exp and runs in tables.")
    # These params are exclusive to each other
    param_group_ls = parser_ls.add_mutually_exclusive_group(required=False)
    param_group_ls.add_argument(
        "-pe",
        "--projs-exps",
        "--projs_exps",
        action="store_true",
        default=True,  # make this as the default
        help="Display all projs and exps.",
    )
    param_group_ls.add_argument("-p", "--projs", action="store_true", help="Display all projs.")
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
        help="Display all runs of the exp of the proj specified by two IDs (proj_ID exp_ID).",
    )
    parser_ls.set_defaults(func=ls)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>>>>>> Subcommand: rm >>>>>>>>>>>>>>>>>>>>>>>>>>
    parser_rm = subparsers.add_parser("rm", help="Remove proj / exp / run.")
    param_group_rm = parser_rm.add_mutually_exclusive_group(required=True)
    param_group_rm.add_argument("-i", "--id", type=str, help="ID of the proj/exp/run that is going to be removed.")
    param_group_rm.add_argument(
        "-e",
        "--exps-of",
        "--exps_of",
        type=str,
        help="ID of the proj whose exp(s) is going to be removed.",
    )
    param_group_rm.add_argument(
        "-r",
        "--runs-of",
        "--runs_of",
        type=str,
        nargs=2,
        help="IDs (proj_ID exp_ID) of the proj and the exp whose run(s) is going to be removed.",
    )
    param_group_rm.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="Remove all contents in the output directory.",
    )
    parser_rm.set_defaults(func=rm)
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
    exec_parent_parser.add_argument("-n", "--name", type=str, required=True, help="Name of the new run.")
    exec_parent_parser.add_argument(
        "-d", "--desc", type=str, required=False, default="", help="Description of the new run."
    )

    # >>>>>>>>>>>>>> Subcommand: fit >>>>>>>>>>>>>>>
    parser_fit = subparsers.add_parser(
        "fit",
        parents=[exec_parent_parser],
        help="Execute a fit run on the dataset's fit split.",
    )
    parser_fit.set_defaults(func=partial(_exec, job_type="fit"))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>> Subcommand: resume-fit >>>>>>>>>>>
    parser_resume_fit = subparsers.add_parser("resume-fit", help="Resume an interrupted fit run.")
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

    # >>>>>>>>>>> Subcommand: test >>>>>>>>>>>
    parser_test = subparsers.add_parser(
        "test",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a test run on the dataset's test split.",
    )
    parser_test.set_defaults(func=partial(_exec, job_type="test"))

    # >>>>>>>>>>> Subcommand: predict >>>>>>>>>>>
    parser_predict = subparsers.add_parser(
        "predict",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a predict run.",
    )
    parser_predict.set_defaults(func=partial(_exec, job_type="predict"))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    return parser.parse_args()


def init(args: argparse.Namespace):
    output_dir = CUBE_CONFIGS["output_dir"]

    proj = EntityFactory.get_proj_instance(name=args.proj_name, desc=args.proj_desc, output_dir=output_dir)
    DAOFactory.get_proj_dao().insert_entry(proj)
    exp = EntityFactory.get_exp_instance(
        name=args.exp_name,
        desc=args.exp_desc,
        output_dir=output_dir,
        proj_id=proj.global_id,
    )
    DAOFactory.get_exp_dao().insert_entry(exp)


def add_exp(args: argparse.Namespace):
    exp = EntityFactory.get_exp_instance(
        name=args.name,
        desc=args.desc,
        output_dir=CUBE_CONFIGS["output_dir"],
        proj_id=args.proj_id,
    )
    DAOFactory.get_exp_dao().insert_entry(exp)


def _draw_table(rich_table, items, prompt_on_empty):
    if len(items) > 0:
        for column_name in items[0]:
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
            rich_table.add_column(outer_column_name, overflow="fold", ratio=width_ratios[idx])

        for item in items:
            inner_table = Table(show_header=True, header_style="bold green")
            if len(item[inner_table_key]) > 0:
                # Add header to the inner table
                for idx, inner_column_name in enumerate(item[inner_table_key][0].keys()):
                    inner_table.add_column(inner_column_name, overflow="fold", ratio=width_ratios[idx])
                # Add rows to the inner table
                for exp in item[inner_table_key]:
                    inner_table.add_row(*list(exp.values()))

            # Add rows to the outer table
            rich_table.add_row(*list(item.values())[:-1], Columns([inner_table]))
    else:
        print(prompt_on_empty)


def ls(args: argparse.Namespace):
    output_dir = CUBE_CONFIGS["output_dir"]
    console = Console()
    get_table = partial(Table, show_header=True, header_style="bold blue", width=console.width)

    # Print all projects
    if args.projs:
        proj_dao = DAOFactory.get_proj_dao()
        table = get_table(title=f'All Projects in "{output_dir}"')
        _draw_table(
            table,
            proj_dao.get_projects(output_dir),
            f'There is no project in "{output_dir}".',
        )

    elif args.exps_runs_of:
        exp_dao = DAOFactory.get_exp_dao()
        table = get_table(title=f'All Exps and Runs of Proj "{args.exps_runs_of}" in "{output_dir}"')
        exps_runs = exp_dao.get_all_exps_runs(output_dir, proj_id=args.exps_runs_of)
        _draw_nested_table(
            table,
            exps_runs,
            inner_table_key="Runs",
            prompt_on_empty=f'There is no exp/run of proj "{args.exps_runs_of}" in {output_dir}.',
        )

    # Print all exps of the specified proj
    elif args.exps_of:
        exp_dao = DAOFactory.get_exp_dao()
        table = get_table(title=f'All Exps of Proj "{args.exps_of}" in "{output_dir}"')
        _draw_table(
            table,
            exp_dao.get_exps_of(output_dir, proj_id=args.exps_of),
            f'There is no exp of proj "{args.exps_of}" in "{output_dir}".',
        )

    # Print all runs of the specified exp of the specified proj
    elif args.runs_of:
        run_dao = DAOFactory.get_run_dao()
        table = get_table(
            title=f'All Runs of exp "{args.runs_of[1]}" of ' f'proj "{args.runs_of[0]}" in "{output_dir}"'
        )
        _draw_table(
            table,
            run_dao.get_runs_of(output_dir, proj_id=args.runs_of[0], exp_id=args.runs_of[1]),
            f'There is no run of exp "{args.runs_of[1]}" of ' f'proj "{args.runs_of[0]}" in "{output_dir}".',
        )

    # Print all projects and exps (default behaviour of `ls`)
    else:
        proj_dao = DAOFactory.get_proj_dao()
        table = get_table(title=f'All Projects and Exps in "{output_dir}"')
        projects_exps = proj_dao.get_all_projects_exps(output_dir)
        _draw_nested_table(
            table,
            projects_exps,
            inner_table_key="Exps",
            prompt_on_empty=f"There is no proj/exp in {output_dir}.",
        )

    console.print(table)


def _confirm_rm(message: str = "") -> bool:
    reply = input(f"{message}, are you sure? [y/n]")
    if reply.lower() == "y":
        return True
    return False


def rm(args: argparse.Namespace):  # noqa: C901
    output_dir = CUBE_CONFIGS["output_dir"]
    proj_dao = DAOFactory.get_proj_dao()
    exp_dao = DAOFactory.get_exp_dao()
    run_dao = DAOFactory.get_run_dao()

    # Remove the entity with specific ID
    if args.id:
        not_found = True
        for proj_dict in proj_dao.get_projects(output_dir):
            if proj_dict["Proj ID"] == args.id:
                if not _confirm_rm(f'Proj "{args.id}" and its all contents will be removed'):
                    return
                proj = proj_dao.get_proj_from_id(output_dir, args.id)
                proj_dao.remove_entry(proj)
                return

            for exp_dict in exp_dao.get_exps_of(output_dir, proj_dict["Proj ID"]):
                if exp_dict["Exp ID"] == args.id:
                    if not _confirm_rm(f'Exp "{args.id}" and its all contents will be removed'):
                        return
                    exp = exp_dao.get_exp_from_id(output_dir, proj_dict["Proj ID"], args.id)
                    exp_dao.remove_entry(exp)
                    return

                for run_dict in run_dao.get_runs_of(output_dir, proj_dict["Proj ID"], exp_dict["Exp ID"]):
                    if run_dict["Run ID"] == args.id:
                        if not _confirm_rm(f'Run "{args.id}" will be removed'):
                            return
                        run = run_dao.get_run_from_id(output_dir, proj_dict["Proj ID"], exp_dict["Exp ID"], args.id)
                        run_dao.remove_entry(run)
                        return
        if not_found:
            warnings.warn(f'There is no any proj/exp/run with ID "{args.id}" found, nothing done.')

    # Remove all exps of in a proj
    elif args.exps_of:
        try:
            exps = exp_dao.get_exps_of(output_dir, args.exps_of)
        except FileNotFoundError:
            warnings.warn(f'There is no proj with ID "{args.exps_of}", nothing done.')
        else:
            if not _confirm_rm(f'Exp(s) of proj "{args.exps_of}" will be removed'):
                return
            for exp_dict in exps:
                exp = exp_dao.get_exp_from_id(output_dir, args.exps_of, exp_dict["Exp ID"])
                exp_dao.remove_entry(exp)

    # Remove all runs of an exp of a proj
    elif args.runs_of:
        try:
            runs = run_dao.get_runs_of(output_dir, args.runs_of[0], args.runs_of[1])
        except (FileNotFoundError, KeyError):
            warnings.warn(f'There is no proj & exp with IDs "{args.runs_of[0]}" "{args.runs_of[1]}", nothing done.')
        else:
            if not _confirm_rm(f'Run(s) of exp "{args.runs_of[0]}" of proj "{args.runs_of[1]}" will be removed'):
                return
            for run_dict in runs:
                run = run_dao.get_run_from_id(output_dir, args.runs_of[0], args.runs_of[1], run_dict["Run ID"])
                run_dao.remove_entry(run)

    elif args.all:
        if not _confirm_rm(f'All contents in "{output_dir}" will be removed'):
            return
        shutil.rmtree(output_dir)


def _exec(args: argparse.Namespace, job_type: JOB_TYPES_T):  # noqa: C901
    root_config = get_root_config_instance(args.config_file)
    output_dir = CUBE_CONFIGS["output_dir"]

    run_dao = DAOFactory.get_run_dao()
    if os.environ.get("CUBE_RUN_ID", None) is None:
        # When resuming fit, the run should resume from the original.
        if job_type == "resume-fit":
            proj_id, exp_id, run_id = run_dao.parse_ids_from_ckpt_path(args.resume_from)
            run = run_dao.get_run_from_id(output_dir, proj_id=proj_id, exp_id=exp_id, run_id=run_id)
            run.is_resuming = True

        # For any other run, a new one should be created.
        else:
            run = EntityFactory.get_run_instance(
                name=args.name,
                desc=args.desc,
                output_dir=output_dir,
                proj_id=args.proj_id,
                exp_id=args.exp_id,
                job_type=job_type,
            )
            run_dao.insert_entry(run)
            run.is_resuming = False

        dump_run(run, output_dir, run.global_id)
        os.environ["CUBE_RUN_ID"] = run.global_id

    else:
        loaded_run = load_run(output_dir, os.environ["CUBE_RUN_ID"])
        run = loaded_run

    CUBE_CONTEXT["run"] = run

    # Set up the task module and the data module.
    root_config.setup_task_data_modules()
    # Set up the trainer(s) for the current run.
    root_config.setup_runners(run=run)

    root_config.callbacks.on_run_start()
    if job_type == "fit":
        root_config.fit_runner.fit(
            model=root_config.task_module,
            datamodule=root_config.data_module,
        )

    elif job_type == "resume-fit":
        root_config.fit_runner.fit(
            root_config.task_module,
            root_config.data_module,
            ckpt_path=args.resume_from,
        )

    elif job_type in ("validate", "test", "predict"):
        if args.loaded_ckpt is not None:
            # Load the specified checkpoint
            loaded_task_module = root_config.task_module.load_checkpoint(args.loaded_ckpt)
            root_config.task_module = loaded_task_module  # update the task module

        run_dao.set_extra_data(run, loaded_ckpt=args.loaded_ckpt)  # save the loaded ckpt info
        if job_type == "validate":
            root_config.validate_runner.validate(
                root_config.task_module,
                root_config.data_module,
            )
        elif job_type == "test":
            root_config.test_runner.test(
                root_config.task_module,
                root_config.data_module,
            )
        else:
            root_config.predict_runner.predict(
                root_config.task_module,
                root_config.data_module,
            )

    else:
        raise ValueError(f"Unrecognized job type: {job_type}!")

    root_config.callbacks.on_run_end()


def main():
    output_dir = None
    try:
        global CUBE_CONFIGS
        CUBE_CONFIGS = parse_cube_configs()
        output_dir = CUBE_CONFIGS["output_dir"]

        if not osp.exists(output_dir):
            os.makedirs(output_dir)

        args = _parse_args()
        args.func(args)
    except Exception as e:
        raise e
    finally:
        run_id = os.environ.get("CUBE_RUN_ID", None)
        if output_dir is not None and run_id is not None:
            remove_run(output_dir, run_id)


if __name__ == "__main__":
    main()
