from cube_dl import __version__
from cube_dl.utils import get_bold_str


def add_subparser_version(subparsers):
    subparser = subparsers.add_parser("version", help="Display the version of cube-dl.")
    subparser.set_defaults(func=print_version, needs_cube_env_check=False)


def print_version(*args, **kwargs):
    print(f"Version of cube-dl: {get_bold_str(__version__)}")
