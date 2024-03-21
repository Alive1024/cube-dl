import os
import os.path as osp
import pickle

from .entities import Run

_CUR_RUN_PREFIX = "run_"


def _get_run_path(dst_dir: str, run_id: str):
    return osp.join(dst_dir, _CUR_RUN_PREFIX + run_id)


def remove_run(dst_dir: str, run_id: str):
    cur_run_path = _get_run_path(dst_dir, run_id)
    if osp.exists(cur_run_path):
        os.remove(cur_run_path)


def load_run(dst_dir: str, run_id: str) -> Run | None:
    cur_run_path = _get_run_path(dst_dir, run_id)
    if not osp.exists(cur_run_path):
        return None

    with open(cur_run_path, "rb") as f:
        return pickle.load(f)


def dump_run(run: Run, dst_dir: str, run_id: str):
    with open(_get_run_path(dst_dir, run_id), "wb") as f:
        pickle.dump(run, f)
