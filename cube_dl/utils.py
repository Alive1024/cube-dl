import os
import os.path as osp

import tomli


class Colors:
    HEADER = "\033[95m"
    OK_BLUE = "\033[94m"
    OK_CYAN = "\033[96m"
    OK_GREEN = "\033[92m"
    WARNING = "\033[93m"  # yellow
    FAIL = "\033[91m"
    END_C = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def get_warning_colored_str(s: str) -> str:
    return Colors.WARNING + s + Colors.END_C


def get_fail_colored_str(s: str) -> str:
    return Colors.FAIL + s + Colors.END_C


def get_ok_green_colored_str(s: str) -> str:
    return Colors.OK_GREEN + s + Colors.END_C


def parse_cube_configs(config_path: str = osp.join(os.getcwd(), "pyproject.toml")) -> dict:
    """Parse configurations from configuration file (default to pyproject.toml)."""

    with open(config_path, "rb") as f:
        configs = tomli.load(f)

    if "tool" not in configs:
        raise RuntimeError(
            get_fail_colored_str(
                f'There is no section named "tool" in "{config_path}", please specify it to configure a '
                "cube-dl project. Or the current directory is not intended for a cube-dl project."
            )
        )

    if "cube_dl" not in configs["tool"]:
        raise RuntimeError(
            get_fail_colored_str(
                f'There is no section named "cube_dl" in "tool in "{config_path}", please specify it. '
                "Or the current directory is not intended for a cube-dl project."
            )
        )

    return configs["tool"]["cube_dl"]
