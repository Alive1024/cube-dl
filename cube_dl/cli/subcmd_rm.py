"""Subcommand: rm."""
import shutil
import warnings
from argparse import Namespace
from copy import deepcopy

from cube_dl import CUBE_CONFIGS
from cube_dl.c3lyr import DAOFactory


def add_subparser_rm(subparsers):
    subparser = subparsers.add_parser("rm", help="Remove proj / exp / run.")
    param_group = subparser.add_mutually_exclusive_group(required=True)
    param_group.add_argument(
        "-i", "--id", type=str, nargs="+", help="ID(s) of the proj/exp/run that is going to be removed."
    )
    param_group.add_argument(
        "-e",
        "--exps-of",
        "--exps_of",
        type=str,
        help="ID of the proj whose exp(s) is going to be removed.",
    )
    param_group.add_argument(
        "-r",
        "--runs-of",
        "--runs_of",
        type=str,
        nargs=2,
        help="IDs (proj_ID exp_ID) of the proj and the exp whose run(s) is going to be removed.",
    )
    param_group.add_argument(
        "-a",
        "--all",
        action="store_true",
        default=False,
        help="Remove all contents in the output directory.",
    )
    subparser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Automatic yes to all confirmation.",
    )
    subparser.set_defaults(func=rm)


def _confirm_rm(message: str = "", yes: bool = False) -> bool:
    if yes:
        return True
    reply = input(f"{message}, proceeded? [y/n] ")
    if reply.lower() == "y":
        return True
    return False


def rm(args: Namespace):  # noqa: C901
    output_dir = CUBE_CONFIGS["output_dir"]
    proj_dao = DAOFactory.get_proj_dao()
    exp_dao = DAOFactory.get_exp_dao()
    run_dao = DAOFactory.get_run_dao()

    # Remove the entity with specific ID(s)
    if args.id:
        entity_ids = deepcopy(args.id)

        for proj_dict in proj_dao.get_projects(output_dir):
            if proj_dict["ID"] in entity_ids:
                if _confirm_rm(f'Proj "{proj_dict["ID"]}" and its all contents will be removed', yes=args.yes):
                    proj = proj_dao.get_proj_from_id(output_dir, proj_dict["ID"])
                    proj_dao.remove_entry(proj)

                entity_ids.remove(proj_dict["ID"])
                if len(entity_ids) == 0:
                    return

            for exp_dict in exp_dao.get_exps_of(output_dir, proj_dict["ID"]):
                if exp_dict["ID"] in entity_ids:
                    if _confirm_rm(f'Exp "{exp_dict["ID"]}" and its all contents will be removed', yes=args.yes):
                        exp = exp_dao.get_exp_from_id(output_dir, proj_dict["ID"], exp_dict["ID"])
                        exp_dao.remove_entry(exp)

                    entity_ids.remove(exp_dict["ID"])
                    if len(entity_ids) == 0:
                        return

                for run_dict in run_dao.get_runs_of(output_dir, proj_dict["ID"], exp_dict["ID"]):
                    if run_dict["ID"] in entity_ids:
                        if _confirm_rm(f'Run "{run_dict["ID"]}" will be removed', yes=args.yes):
                            run = run_dao.get_run_from_id(output_dir, proj_dict["ID"], exp_dict["ID"], run_dict["ID"])
                            run_dao.remove_entry(run)

                        entity_ids.remove(run_dict["ID"])
                        if len(entity_ids) == 0:
                            return

        if len(entity_ids) != 0:
            warnings.warn(f'There is no any proj/exp/run with ID "{", ".join(entity_ids)}" found.')

    # Remove all exps of in a proj
    elif args.exps_of:
        try:
            exps = exp_dao.get_exps_of(output_dir, args.exps_of)
        except FileNotFoundError:
            warnings.warn(f'There is no proj with ID "{args.exps_of}", nothing done.')
        else:
            if not _confirm_rm(f'Exp(s) of proj "{args.exps_of}" will be removed', yes=args.yes):
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
            if not _confirm_rm(
                f'Run(s) of exp "{args.runs_of[0]}" of proj "{args.runs_of[1]}" will be removed', yes=args.yes
            ):
                return
            for run_dict in runs:
                run = run_dao.get_run_from_id(output_dir, args.runs_of[0], args.runs_of[1], run_dict["ID"])
                run_dao.remove_entry(run)

    elif args.all:
        if not _confirm_rm(f'All contents in "{output_dir}" will be removed', yes=args.yes):
            return
        shutil.rmtree(output_dir)
