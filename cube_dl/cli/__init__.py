"""Package for implementing command line interface."""
import argparse

from .subcmd_add_exp import add_subparser_add_exp
from .subcmd_ls import add_subparser_ls
from .subcmd_new import add_subparser_new
from .subcmd_rm import add_subparser_rm
from .subcmd_start import add_subparser_start
from .subcmds_exec import add_subparser_exec

__all__ = ["parse_args"]


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    add_subparser_start(subparsers)
    add_subparser_new(subparsers)
    add_subparser_add_exp(subparsers)
    add_subparser_ls(subparsers)
    add_subparser_rm(subparsers)
    add_subparser_exec(subparsers)

    return parser.parse_args()
