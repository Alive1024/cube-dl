"""Subcommands: fit, resume-fit, validate, test and predict."""
import argparse
import os
from argparse import Namespace
from functools import partial
from typing import Literal
from warnings import warn

from cube_dl import CUBE_CONFIGS
from cube_dl.c3lyr import DAOFactory, EntityFactory, dump_run, load_run
from cube_dl.config_sys import get_root_config_instance
from cube_dl.core import CUBE_CONTEXT

JOB_TYPES_T = Literal["fit", "resume-fit", "validate", "test", "predict"]


def add_subparser_exec(subparsers):
    # Create a parent subparser containing common arguments.
    # Ref: https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers
    exec_parent_parser = argparse.ArgumentParser(add_help=False)
    exec_parent_parser.add_argument(
        "-c",
        "--config-file",
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    exec_parent_parser.add_argument(
        "-p",
        "--proj-id",
        "--proj_id",
        type=str,
        required=True,
        help="ID of the proj ID that the new run belongs to.",
    )
    exec_parent_parser.add_argument(
        "-e",
        "--exp-id",
        "--exp_id",
        type=str,
        required=True,
        help="ID of the exp that the new run belongs to.",
    )
    exec_parent_parser.add_argument("-n", "--name", type=str, required=True, help="Name of the new run.")
    exec_parent_parser.add_argument(
        "-d", "--desc", type=str, required=False, default="", help="Description of the new run."
    )

    # >>>>>>>>>>>>>> Subcommand: fit >>>>>>>>>>>>>>>
    parser_fit = subparsers.add_parser(
        "fit",
        parents=[exec_parent_parser],
        help="Execute a fit run on the dataset's fit split.",
    )
    parser_fit.set_defaults(func=partial(execute, job_type="fit"))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>> Subcommand: resume-fit >>>>>>>>>>>
    parser_resume_fit = subparsers.add_parser("resume-fit", help="Resume an interrupted fit run.")
    parser_resume_fit.add_argument(
        "-c",
        "--config-file",
        "--config_file",
        type=str,
        required=True,
        help="Path to the config file.",
    )
    parser_resume_fit.add_argument(
        "-r",
        "--resume-from",
        "--resume_from",
        type=str,
        required=True,
        help="File path to the checkpoint where resumes.",
    )
    parser_resume_fit.set_defaults(func=partial(execute, job_type="resume-fit"))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # >>>>>>>>>>>>>>>>>>>>>> Subcommands: validate, test and predict >>>>>>>>>>>>>>>>>>>>>>
    # Common parent parser.
    validate_test_predict_parent_parser = argparse.ArgumentParser(add_help=False)
    validate_test_predict_parent_parser.add_argument(
        "-lc",
        "--loaded-ckpt",
        "--loaded_ckpt",
        type=str,
        help='File path of the model checkpoint to be loaded. Use an empty string "" to explicitly indicate '
        "you are going to conduct validate/test/predict using the initialized model without "
        "loading any weights).",
    )
    # >>>>>>>>>>> Subcommand: validate >>>>>>>>>>>
    parser_validate = subparsers.add_parser(
        "validate",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a validate run on the dataset's validate split.",
    )
    parser_validate.set_defaults(func=partial(execute, job_type="validate"))

    # >>>>>>>>>>> Subcommand: test >>>>>>>>>>>
    parser_test = subparsers.add_parser(
        "test",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a test run on the dataset's test split.",
    )
    parser_test.set_defaults(func=partial(execute, job_type="test"))

    # >>>>>>>>>>> Subcommand: predict >>>>>>>>>>>
    parser_predict = subparsers.add_parser(
        "predict",
        parents=[exec_parent_parser, validate_test_predict_parent_parser],
        help="Execute a predict run.",
    )
    parser_predict.set_defaults(func=partial(execute, job_type="predict"))


def execute(args: Namespace, job_type: JOB_TYPES_T):  # noqa: C901
    root_config = get_root_config_instance(args.config_file)
    output_dir = CUBE_CONFIGS["output_dir"]

    run_dao = DAOFactory.get_run_dao()
    if os.environ.get("CUBE_RUN_ID", None) is None:
        # When resuming fit, the run should resume from the original.
        if job_type == "resume-fit":
            proj_id, exp_id, run_id = run_dao.parse_ids_from_ckpt_path(args.resume_from)
            run = run_dao.get_run_from_id(output_dir, proj_id=proj_id, exp_id=exp_id, run_id=run_id)
            run.is_resuming = True

        # For any other run, a new one should be created.
        else:
            run = EntityFactory.get_run_instance(
                name=args.name,
                desc=args.desc,
                output_dir=output_dir,
                proj_id=args.proj_id,
                exp_id=args.exp_id,
                job_type=job_type,
            )
            run_dao.insert_entry(run)
            run.is_resuming = False

        dump_run(run, output_dir, run.global_id)
        os.environ["CUBE_RUN_ID"] = run.global_id

    else:
        loaded_run = load_run(output_dir, os.environ["CUBE_RUN_ID"])
        run = loaded_run

    CUBE_CONTEXT["run"] = run

    # Set up the task module and the data module.
    root_config.setup_task_data_modules()
    # Set up the trainer(s) for the current run.
    root_config.setup_runners(run=run)

    root_config.callbacks.on_run_start()
    if job_type == "fit":
        root_config.fit_runner.fit(
            model=root_config.task_module,
            datamodule=root_config.data_module,
        )

    elif job_type == "resume-fit":
        root_config.fit_runner.fit(
            root_config.task_module,
            root_config.data_module,
            ckpt_path=args.resume_from,
        )

    elif job_type in ("validate", "test", "predict"):
        if args.loaded_ckpt.strip() != "":
            # Load the specified checkpoint
            loaded_task_module = root_config.task_module.load_checkpoint(args.loaded_ckpt)
            root_config.task_module = loaded_task_module  # update the task module
        else:
            warn(
                'The argument "-lc" is empty, you are going to conduct validate/test/predict using the '
                "initialized model without loading any weights)."
            )

        run_dao.set_extra_data(run, loaded_ckpt=args.loaded_ckpt)  # save the loaded ckpt info
        if job_type == "validate":
            root_config.validate_runner.validate(
                root_config.task_module,
                root_config.data_module,
            )
        elif job_type == "test":
            root_config.test_runner.test(
                root_config.task_module,
                root_config.data_module,
            )
        else:
            root_config.predict_runner.predict(
                root_config.task_module,
                root_config.data_module,
            )

    else:
        raise ValueError(f"Unrecognized job type: {job_type}!")

    root_config.callbacks.on_run_end()
