import os
import os.path as osp
import pickle
import time
import warnings

from .entities import Run

CUR_RUN_FILENAME = "cur_run"
TIMEOUT_LIMIT = 60  # in seconds


def remove_cur_run(dst_dir: str):
    cur_run_path = osp.join(osp.join(dst_dir, CUR_RUN_FILENAME))
    if osp.exists(cur_run_path):
        os.remove(cur_run_path)


def try_to_load_run(dst_dir: str) -> Run | None:
    cur_run_path = osp.join(osp.join(dst_dir, CUR_RUN_FILENAME))
    if not osp.exists(cur_run_path):
        return None

    if time.time() - osp.getctime(cur_run_path) > TIMEOUT_LIMIT:
        warnings.warn("Suspected stale run detected, probably caused by abnormal exit last time, deleting it.")
        os.remove(cur_run_path)
        return None

    with open(cur_run_path, "rb") as f:
        return pickle.load(f)


def dump_run(run: Run, dst_dir: str):
    with open(osp.join(dst_dir, CUR_RUN_FILENAME), "wb") as f:
        pickle.dump(run, f)
