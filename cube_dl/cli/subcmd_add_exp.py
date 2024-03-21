"""Subcommand: add-exp."""
from argparse import Namespace

from cube_dl import CUBE_CONFIGS
from cube_dl.c3lyr import DAOFactory, EntityFactory


def add_subparser_add_exp(subparsers):
    subparser = subparsers.add_parser("add-exp", help="Create a new exp within specified proj.")
    subparser.add_argument(
        "-p",
        "--proj-id",
        "--proj_id",
        type=str,
        required=True,
        help="ID of the proj that the new exp belongs to.",
    )
    subparser.add_argument(
        "-n",
        "--name",
        type=str,
        required=True,
        help="Name of the new exp.",
    )
    subparser.add_argument(
        "-d",
        "--desc",
        type=str,
        required=False,
        default="",
        help="(Optional) Description of the new exp.",
    )
    subparser.set_defaults(func=add_exp)


def add_exp(args: Namespace):
    exp = EntityFactory.get_exp_instance(
        name=args.name,
        desc=args.desc,
        output_dir=CUBE_CONFIGS["output_dir"],
        proj_id=args.proj_id,
    )
    DAOFactory.get_exp_dao().insert_entry(exp)
