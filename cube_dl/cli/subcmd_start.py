"""Subcommand: start."""
import os
import os.path as osp
from argparse import Namespace
from time import sleep

import requests
from rich.console import Console
from rich.table import Table

from cube_dl.utils import get_fail_colored_str, get_ok_green_colored_str, parse_cube_configs

from .utils import plot_table


def add_subparser_start(subparsers):
    subparser = subparsers.add_parser(
        "start", help='Start from a "starter". (Only starters from public Github repo are supported for now.)'
    )
    subparser.add_argument("-l", "--list", action="store_true", help="List all available starters.")
    subparser.add_argument(
        "-o",
        "--owner",
        type=str,
        default="Alive1024",
        help="Owner(GitHub username) of the GitHub repo that the starter is from. "
        'Default to "Alive1024" (original author).',
    )
    subparser.add_argument(
        "-r",
        "--repo",
        type=str,
        default="cube-dl",
        help='Name of the GitHub repo that the starter is from. Default to "cube-dl".',
    )
    subparser.add_argument("-p", "--path", type=str, help="Path to the starter (a directory).")
    subparser.add_argument(
        "-d",
        "--dest",
        type=str,
        default=None,
        help="Destination directory to store the starter. Default to the current working directory.",
    )

    subparser.set_defaults(func=start)


def _list_available_starters() -> Table:
    """List available starters."""
    api_url = "https://raw.githubusercontent.com/Alive1024/cube-dl/main/starters/starter-registry.json"
    resp = requests.get(api_url)
    try:
        starters = resp.json()
        return plot_table(title="Available Starters", items=starters, prompt_on_empty="No Available Starters")
    except requests.exceptions.JSONDecodeError as e:
        print(get_fail_colored_str(str(resp)))
        raise e


def _download_file_from_url(url, save_path):
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)


def _download_github_directory(owner, repo, target_dir, save_dir):
    """Download a directory from the given GitHub URL using GitHub's API."""
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{target_dir}"
    contents = requests.get(api_url).json()

    if isinstance(contents, dict) and "message" in contents and contents["message"] == "Not Found":
        raise RuntimeError("File(s) NOT FOUND for given information, please check.")

    for item in contents:
        name = item["name"]
        if item["type"] == "file":
            file_url = item["download_url"]
            _download_file_from_url(file_url, osp.join(save_dir, name))
            sleep(0.1)  # time delay to avert GitHub's API rate limit
        # Download files in subdirectories recursively
        elif item["type"] == "dir":
            subdir = osp.join(save_dir, name)
            if not osp.exists(subdir):
                os.mkdir(subdir)
            _download_github_directory(owner, repo, f"{target_dir}/{name}", subdir)


def start(args: Namespace):
    if args.list:
        console = Console()
        console.print(_list_available_starters())
    else:
        save_dir = os.getcwd() if args.dest is None else args.dest
        print("Downloading files...")
        _download_github_directory(
            owner=args.owner,
            repo=args.repo,
            target_dir=args.path,
            save_dir=save_dir,
        )
        print(get_ok_green_colored_str("Downloading completed."))

        if not osp.exists(osp.join(save_dir, "pyproject.toml")):
            raise RuntimeError(
                get_fail_colored_str(
                    "It seems like the directory downloaded is not a standard "
                    'cube-dl starter, as "pyproject.toml" is NOT FOUND.'
                )
            )
        else:
            # Trying to parse cube configs
            parse_cube_configs()
