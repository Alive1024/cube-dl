import os
import os.path as osp
import sys

from cube_dl import CUBE_CONFIGS
from cube_dl.c3lyr import remove_run
from cube_dl.cli import parse_args
from cube_dl.cli.subcmd_start import start
from cube_dl.utils import get_warning_colored_str, parse_cube_configs

# Add current working directory to PYTHONPATH  # noqa
sys.path.insert(0, os.getcwd())


def main():
    output_dir = None
    try:
        args = parse_args()
        if args.func == start:
            args.func(args)
        elif not osp.exists(osp.join(os.getcwd(), "pyproject.toml")):
            print(
                get_warning_colored_str(
                    "The current working directory is not a cube-dl project "
                    '(as "pyproject.toml" is NOT FOUND). \n'
                    "Please run `cube start` with necessary arguments to get started "
                    "if you want to use this directory to store a cube-dl project."
                )
            )
        else:
            CUBE_CONFIGS.update(parse_cube_configs())
            output_dir = CUBE_CONFIGS["output_dir"]
            if not osp.exists(output_dir):
                os.makedirs(output_dir)
                # Add a .gitignore to allow git to track project record files and .gitkeep in exp directories
                with open(osp.join(output_dir, ".gitignore"), "w") as f:
                    f.write("*\n!*/\n!.gitignore\n!*.json\n!.gitkeep\n")
            args.func(args)
    except Exception as e:
        raise e
    finally:
        run_id = os.environ.get("CUBE_RUN_ID", None)
        if output_dir is not None and run_id is not None:
            remove_run(output_dir, run_id)


if __name__ == "__main__":
    main()
