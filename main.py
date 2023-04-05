import os.path as osp
import argparse
import importlib
import re

import pytorch_lightning as pl

from root_config import RootConfig
from entities import Project, Experiment, Run

OUTPUT_DIR = "./outputs"
GLOBAL_SEED = 42

JOB_TYPES = ("fit", "validate", "test", "predict", "tune")


def _check_trainer(trainer: pl.Trainer, job_type: str):
    if not trainer:
        raise ValueError(f"There is no trainer for {job_type}, please specify it in the Config instance.")


def _check_test_predict_before_fit(runs: list, current_stage_idx: int) -> bool:
    """Check whether the "test"/"predict" stages comes before "fit" in the given runs.
    :return: bool. True means there is.
    """
    has_fitted = False
    for stage in runs[:current_stage_idx + 1]:
        if stage == "fit":
            has_fitted = True
        if (stage == "test" or stage == "predict") and (not has_fitted):
            return True
    return False


def _get_root_config_instance(config_file_path, return_getter=False):
    # Remove the part of root directory in an absolute path.
    relative_file_path = config_file_path.replace(osp.split(__file__)[0], '')
    import_path = relative_file_path.replace('\\', '.').replace('/', '.')

    # Remove the opening '.', or importlib will seem it as a relative import and the `package` argument is needed.
    if import_path.startswith('.'):
        import_path = import_path[1:]
    if import_path.endswith(".py"):
        import_path = import_path[:-3]

    try:
        root_config_getter = getattr(importlib.import_module(import_path), RootConfig.CONFIG_GETTER_NAME)
        config_instance: RootConfig = root_config_getter()
    except AttributeError:
        raise AttributeError(f"Expected a function called `{RootConfig.CONFIG_GETTER_NAME}` in your root config file, "
                             f"like `def {RootConfig.CONFIG_GETTER_NAME}(): ...`")
    return (config_instance, root_config_getter) if return_getter else config_instance


def _init(args):
    proj = Project(name=args.proj_name, desc=args.proj_desc, output_dir=OUTPUT_DIR)
    Experiment(name=args.exp_name, desc=args.exp_desc, proj_id=proj.global_id, output_dir=OUTPUT_DIR)


def _add_exp(args):
    Experiment(name=args.exp_name, desc=args.exp_desc, proj_id=args.proj_id, output_dir=OUTPUT_DIR)


def _exec(args):
    # ============================ Checking Arguments ============================
    runs: list = re.split(r"[\s,]+", args.runs)
    for job_type in runs:
        if job_type not in JOB_TYPES:
            raise ValueError(f"Unrecognized job_type: `{job_type}`, please choose from {JOB_TYPES}.")

    if ("fit" not in runs) and (args.fit_resumes_from is not None):
        raise ValueError("There is no `fit` in the runs but `--fit-resumes-from` are provided.")

    # If there is test/predict before fit in the runs, `--test-predict-ckpt` must be explicitly provided
    if _check_test_predict_before_fit(runs, current_stage_idx=len(runs) - 1) and args.test_predict_ckpt == '':
        raise ValueError("There is test/predict before fit in the runs, "
                         "please provide the argument `--test-predict-ckpt`. Run `python main.py -h` to learn more.")
    # ================================================================================

    pl.seed_everything(GLOBAL_SEED)
    root_config_instance, root_config_getter = _get_root_config_instance(args.config_file, return_getter=True)

    # ============================ Archive the Config Files ============================
    if not args.fit_resumes_from:
        archived_configs_dir = Experiment.get_archived_configs_dir(proj_id=args.proj_id, exp_id=args.exp_id,
                                                                   output_dir=OUTPUT_DIR)
        start_run_id = Run.get_next_id(proj_dir=osp.join(OUTPUT_DIR,
                                                         Project.get_proj_name_from_id(OUTPUT_DIR, args.proj_id)))
        # root_config_instance.archive_config_into_dir_or_zip(root_config_getter,
        #                                                     archived_configs_dir=archived_configs_dir,
        #                                                     start_run_id=start_run_id,
        #                                                     end_start_id=start_run_id + len(runs) - 1)
        root_config_instance.archive_config_into_single(root_config_getter,
                                                        archived_configs_dir=archived_configs_dir,
                                                        start_run_id=start_run_id,
                                                        end_start_id=start_run_id + len(runs) - 1)
    # ==================================================================================

    # ============================ Executing the Runs ============================
    # Set up the task wrapper and the data wrapper.
    root_config_instance.setup_wrappers()

    for idx, job_type in enumerate(runs):
        print("\n\n", "*" * 35, f"Launching {job_type}, ({idx + 1}/{len(runs)})", "*" * 35, '\n')

        run = Run(name=args.name, desc=args.desc, proj_id=args.proj_id, exp_id=args.exp_id,
                  job_type=job_type, output_dir=OUTPUT_DIR, resume_from=args.fit_resumes_from)

        # Set up the trainer(s) for the current run.
        root_config_instance.setup_trainer(logger_arg=args.logger, run=run)

        if job_type == "fit":
            _check_trainer(root_config_instance.fit_trainer, job_type)
            root_config_instance.fit_trainer.fit(
                model=root_config_instance.task_wrapper,
                datamodule=root_config_instance.data_wrapper,
                ckpt_path=args.fit_resumes_from
            )
            # Merge multiple metrics_csv (if any) into "merged_metrics.csv" after fit exits
            # (finishes / KeyboardInterrupt).
            Run.merge_metrics_csv(run.run_dir)
            Run.remove_empty_hparams_yaml(run.run_dir)

        elif job_type == "validate":
            _check_trainer(root_config_instance.validate_trainer, job_type)
            root_config_instance.validate_trainer.validate(
                model=root_config_instance.task_wrapper,
                datamodule=root_config_instance.data_wrapper
            )
        elif job_type == "test":
            _check_trainer(root_config_instance.test_trainer, job_type)

            # If the test job_type appears before fit
            if _check_test_predict_before_fit(runs, idx) and args.test_predict_ckpt != "none":
                # Load the specified checkpoint
                loaded_task_wrapper = root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                    args.test_predict_ckpt,
                    **root_config_instance.task_wrapper.get_init_args()
                )
                root_config_instance.task_wrapper = loaded_task_wrapper  # update the task wrapper

            # Otherwise, do nothing, just keep the model state unchanged.
            root_config_instance.test_trainer.test(
                model=root_config_instance.task_wrapper,
                datamodule=root_config_instance.data_wrapper
            )
        elif job_type == "predict":
            _check_trainer(root_config_instance.predict_trainer, job_type)
            # Same as "test"
            if _check_test_predict_before_fit(runs, idx) and args.test_predict_ckpt != "none":
                loaded_task_wrapper = root_config_instance.task_wrapper.__class__.load_from_checkpoint(
                    args.test_predict_ckpt,
                    **root_config_instance.task_wrapper.get_init_args()
                )
                root_config_instance.task_wrapper = loaded_task_wrapper

            root_config_instance.predict_trainer.predict(
                model=root_config_instance.task_wrapper,
                datamodule=root_config_instance.data_wrapper
            )
        elif job_type == "tune":
            # TODO
            raise NotImplementedError
    # ============================================================================


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Subcommand: init
    parser_init = subparsers.add_parser("init", help="")
    parser_init.add_argument("-pn", "--proj-name", "--proj_name", type=str, required=True,
                             help="")
    parser_init.add_argument("-pd", "--proj-desc", "--proj_desc", type=str, required=True,
                             help="")
    parser_init.add_argument("-en", "--exp-name", "--exp_name", type=str, required=True,
                             help="")
    parser_init.add_argument("-ed", "--exp-desc", "--exp_desc", type=str, required=True,
                             help="")
    parser_init.set_defaults(func=_init)

    # Subcommand: add-exp
    parser_exp = subparsers.add_parser("add-exp", help="Create an exp with an optional name and "
                                                       "a compulsory description.")
    parser_exp.add_argument("-p", "--proj-id", "--proj_id", type=int, required=True,
                            help="")
    parser_exp.add_argument("-en", "--exp-name", "--exp_name", type=str, required=True,
                            help="The name of the experiment, it will be appended to the prefix: exp_{exp_id}.")
    parser_exp.add_argument("-ed", "--exp-desc", "--exp_desc", type=str, required=True,
                            help="A description about the created exp.")
    parser_exp.set_defaults(func=_add_exp)

    # Subcommand: exec
    parser_exec = subparsers.add_parser("exec", help="Execute the run(s)")
    parser_exec.add_argument("-p", "--proj-id", "--proj_id", type=int, required=True,
                             help="")
    parser_exec.add_argument("-e", "--exp-id", "--exp_id", type=int, required=True,
                             help="The experiment id of which the current run(s) belong to, "
                                  "create one at first if not any.")

    parser_exec.add_argument("-c", "--config-file", "--config_file", type=str, required=True,
                             help="Path to the config file")
    parser_exec.add_argument("-r", "--runs", type=str, required=True,
                             help=f"A sequence of run. Choose from {JOB_TYPES} "
                                  "and combine arbitrarily, seperated by comma. e.g. \"tune,fit,test\" ")
    parser_exec.add_argument("-n", "--name", type=str, default=None,
                             help="The name of the current run(s), it will always have a prefix: run_{idx}.")
    parser_exec.add_argument("-d", "--desc", type=str, required=True,
                             help="A description about the current run(s).")

    parser_exec.add_argument("--fit-resumes-from", "--fit_resumes_from", type=str, default=None,
                             help="")
    parser_exec.add_argument("--test-predict-ckpt", "--test_predict_ckpt", type=str, default='',
                             help="This must be explicitly provided if you execute test/predict before fit "
                                  "in current runs. It can be a file path, or \"none\" "
                                  "(this means you are going to take test/predict using the initialized model "
                                  "without training).")
    parser_exec.add_argument("-l", "--logger", type=str, default="CSV",
                             help=f"Choose from {RootConfig.LOGGERS} "
                                  f"and combine arbitrarily, seperated by comma. e.g. \"csv,wandb\" "
                                  f"Or it can be True/False, meaning using the default CSV and "
                                  f"disable logging respectively.")
    parser_exec.set_defaults(func=_exec)

    # Parse args and invoke the corresponding function
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
