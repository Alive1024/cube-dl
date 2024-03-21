"""Subcommand: new."""
from argparse import Namespace

from cube_dl import CUBE_CONFIGS
from cube_dl.c3lyr import DAOFactory, EntityFactory


def add_subparser_new(subparsers):
    subparser = subparsers.add_parser("new", help="Create a pair of new proj and exp.")
    subparser.add_argument(
        "-pn",
        "--proj-name",
        "--proj_name",
        type=str,
        required=True,
        help="Name of the new proj.",
    )
    subparser.add_argument(
        "-pd",
        "--proj-desc",
        "--proj_desc",
        type=str,
        required=False,
        default="",
        help="(Optional) Description of the new proj.",
    )
    subparser.add_argument(
        "-en",
        "--exp-name",
        "--exp_name",
        type=str,
        required=True,
        help="Name of the new exp.",
    )
    subparser.add_argument(
        "-ed",
        "--exp-desc",
        "--exp_desc",
        type=str,
        required=False,
        default="",
        help="(Optional) Description of the new exp.",
    )
    subparser.set_defaults(func=new)


def new(args: Namespace):
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
