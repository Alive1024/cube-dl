import os
import os.path as osp

import tomli


def parse_cube_configs(config_path: str = osp.join(os.getcwd(), "pyproject.toml")) -> dict:
    """Parse configurations from configuration file (default to pyproject.toml)."""
    if not osp.exists(config_path):
        raise RuntimeError(f'Cube configuration file ("pyproject.toml") NOT FOUND in "{config_path}".')

    with open(config_path, "rb") as f:
        configs = tomli.load(f)

    if "tool" not in configs:
        raise RuntimeError(
            f'There is no section named "tool" in "{config_path}", please specify it to configure a Cube project.'
            "Or the current directory is not intended for a Cube project."
        )

    if "cube" not in configs["tool"]:
        raise RuntimeError(f'There is no section named "cube" in "tool in "{config_path}", please specify it.')

    return configs["tool"]["cube"]
