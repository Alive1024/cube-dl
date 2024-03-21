"""Subcommand: ls."""
from argparse import Namespace

from rich.console import Console

from cube_dl import CUBE_CONFIGS
from cube_dl.c3lyr import DAOFactory

from .utils import plot_nested_table, plot_table


def add_subparser_ls(subparsers):
    subparser = subparsers.add_parser("ls", help="Display info about proj, exp and runs in tables.")
    # These params are exclusive to each other
    param_group_ls = subparser.add_mutually_exclusive_group(required=False)
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
    subparser.set_defaults(func=ls)


def ls(args: Namespace):
    output_dir = CUBE_CONFIGS["output_dir"]
    console = Console()

    # Print all projects
    if args.projs:
        proj_dao = DAOFactory.get_proj_dao()
        table = plot_table(
            title=f'All Projects in "{output_dir}"',
            items=proj_dao.get_projects(output_dir),
            prompt_on_empty=f'There is no project in "{output_dir}".',
        )

    elif args.exps_runs_of:
        exp_dao = DAOFactory.get_exp_dao()
        exps_runs = exp_dao.get_all_exps_runs(output_dir, proj_id=args.exps_runs_of)
        table = plot_nested_table(
            title=f'All Exps and Runs of Proj "{args.exps_runs_of}" in "{output_dir}"',
            items=exps_runs,
            inner_table_key="Runs",
            prompt_on_empty=f'There is no exp/run of proj "{args.exps_runs_of}" in {output_dir}.',
        )

    # Print all exps of the specified proj
    elif args.exps_of:
        exp_dao = DAOFactory.get_exp_dao()
        table = plot_table(
            title=f'All Exps of Proj "{args.exps_of}" in "{output_dir}"',
            items=exp_dao.get_exps_of(output_dir, proj_id=args.exps_of),
            prompt_on_empty=f'There is no exp of proj "{args.exps_of}" in "{output_dir}".',
        )

    # Print all runs of the specified exp of the specified proj
    elif args.runs_of:
        run_dao = DAOFactory.get_run_dao()
        table = plot_table(
            title=f'All Runs of exp "{args.runs_of[1]}" of ' f'proj "{args.runs_of[0]}" in "{output_dir}"',
            items=run_dao.get_runs_of(output_dir, proj_id=args.runs_of[0], exp_id=args.runs_of[1]),
            prompt_on_empty=f'There is no run of exp "{args.runs_of[1]}" of '
            f'proj "{args.runs_of[0]}" in "{output_dir}".',
        )

    # Print all projects and exps (default behaviour of `ls`)
    else:
        proj_dao = DAOFactory.get_proj_dao()
        projects_exps = proj_dao.get_all_projects_exps(output_dir)
        table = plot_nested_table(
            title=f'All Projects and Exps in "{output_dir}"',
            items=projects_exps,
            inner_table_key="Exps",
            prompt_on_empty=f"There is no proj/exp in {output_dir}.",
        )

    console.print(table)
