"""
Entities implementing the "three-layer concepts": project, experiment, run, dealing with directories, filenames
"""
import os
import os.path as osp
from datetime import datetime
from abc import abstractmethod
from typing import List, Literal, Union, Optional, Callable
from functools import wraps
import re
import warnings
import shutil
import json


def rank_zero_only(fn: Callable) -> Callable:
    """
    Function that can be used as a decorator to enable a function/method being called only on global rank 0.
    Borrowed from lightning_utilities/core/rank_zero.py
    """
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn


def get_next_id(dst_dir, id_type: Literal["proj", "exp", "run", "metrics_csv"]) -> int:
    """
    Get the global id for proj / exp / run / CSVLogger's metrics.csv.
    :param dst_dir:
    :param id_type:
    :return:
    """
    indices = []

    # Iterate over the output dir and proj dir respectively
    if id_type == "proj" or id_type == "exp":
        for i in os.listdir(dst_dir):
            if i.startswith(id_type + '_'):
                indices.append(int(i.split('_')[1]))

    # Iterate over all the exp dirs within the project directory to find the global run id.
    elif id_type == "run":
        for i in os.listdir(dst_dir):
            if i.startswith("exp_"):
                for j in os.listdir(osp.join(dst_dir, i)):
                    if j.startswith("run_"):
                        indices.append(int(j.split('_')[1]))

    # Iterate within the run dir
    elif id_type == "metrics_csv":
        for filename in os.listdir(dst_dir):
            if filename.startswith("metrics_"):
                indices.append(int(osp.splitext(filename)[0].split('_')[1]))
    return 1 if len(indices) == 0 else (max(indices) + 1)


def get_name_from_id(output_dir, ids: Union[int, List[int]], id_type: Literal["proj", "exp"]):
    if id_type == "proj":
        proj_name = ""
        for fn in os.listdir(output_dir):
            if fn.startswith(f"proj_{ids}"):
                proj_name = fn
        if not proj_name:
            raise RuntimeError("")
        return proj_name

    if id_type == "exp":
        proj_name, exp_name = "", ""
        for fn1 in os.listdir(output_dir):
            if fn1.startswith(f"proj_{ids[0]}"):
                proj_name = fn1
                for fn2 in os.listdir(osp.join(output_dir, fn1)):
                    if fn2.startswith(f"exp_{ids[1]}"):
                        exp_name = fn2
        if not proj_name:
            raise RuntimeError("")
        if not exp_name:
            raise RuntimeError("")
        return proj_name, exp_name


class _EntityBase:
    entity_type: Literal["proj", "exp", "run"]

    def __init__(self, relative_dir: str, name: str, desc: str, record_file_path: str, output_dir: str):
        self.global_id: int = get_next_id(relative_dir, id_type=self.entity_type)

        self.name = f"{self.entity_type}_{self.global_id}_{name}"
        self.desc = desc
        self.record_file_path = record_file_path
        self.output_dir = output_dir
        self.created_time = datetime.now().strftime("%Y-%m-%d (%a) %H:%M:%S")

        self._extra_record_data = {}

    def print_message(self, storage_path):
        print(f"{self.entity_type}: {self.name} created, storage path: {storage_path}")

    # ################# Setter and getter for extra record data, which allows for save custom data.
    def set_extra_record_data(self, **kwargs):
        self._extra_record_data.update(kwargs)
        self._update_record_entry()

    def get_extra_record_data(self, key: str):
        return self._extra_record_data[key]

    # ###############

    @abstractmethod
    def _get_record_dict(self) -> dict:
        raise NotImplementedError

    def _get_record_entry(self) -> dict:
        record_dict = self._get_record_dict()
        record_dict.update(self._extra_record_data)
        return record_dict

    @abstractmethod
    def _write_new_record_entry(self):
        raise NotImplementedError

    @abstractmethod
    def _update_record_entry(self):
        """
        Used when a key-value pair is set, i.e. when `set_extra_record_data` is called.
        """
        raise NotImplementedError


class Project(_EntityBase):
    entity_type = "proj"

    def __init__(self, name: str, desc: str, output_dir: str):
        super().__init__(relative_dir=output_dir, name=name, desc=desc,
                         record_file_path="", output_dir=output_dir)
        self.proj_dir = osp.join(output_dir, self.name)
        os.mkdir(self.proj_dir)
        self.record_file_path = osp.join(self.proj_dir, self.name + ".json")
        self._write_new_record_entry()
        self.print_message(self.proj_dir)

    def _get_record_dict(self) -> dict:
        return {
            "Project ID": self.global_id,
            "Project Name": self.name,
            "Project Description": self.desc,
            "Created Time": self.created_time,
            "Storage Path": self.output_dir,
            "Exps": []
        }

    def _write_new_record_entry(self):
        with open(self.record_file_path, 'w') as f:
            print(self._get_record_dict())
            json.dump(self._get_record_entry(), f, indent=2)

    def _update_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            record.update(self._extra_record_data)
        with open(self.record_file_path, 'w') as f:
            json.dump(record, f, indent=2)


class Experiment(_EntityBase):
    entity_type = "exp"

    def __init__(self, name: str, desc: str, proj_id: int, output_dir: str):
        proj_name = get_name_from_id(output_dir, proj_id, id_type="proj")
        self.proj_dir = osp.join(output_dir, proj_name)

        super().__init__(relative_dir=self.proj_dir, name=name, desc=desc,
                         record_file_path=osp.join(self.proj_dir, proj_name + ".json"),
                         output_dir=output_dir)

        self.exp_dir = osp.join(self.proj_dir, self.name)
        os.mkdir(self.exp_dir)
        os.mkdir(osp.join(self.exp_dir, "archived_configs"))  # the directory for archived configs

        self._write_new_record_entry()
        self.print_message(self.exp_dir)

    @staticmethod
    def get_archived_configs_dir(proj_id: int, exp_id: int, output_dir: str):
        proj_name, exp_name = get_name_from_id(output_dir, [proj_id, exp_id], id_type="exp")
        return osp.join(output_dir, proj_name, exp_name, "archived_configs")

    def _get_record_dict(self) -> dict:
        return {
            "Exp ID": self.global_id,
            "Exp Name": self.name,
            "Exp Description": self.desc,
            "Created Time": self.created_time,
            "Runs": []
        }

    def _write_new_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            record["Exps"].append(self._get_record_entry())
        with open(self.record_file_path, 'w') as f:
            json.dump(record, f, indent=2)

    def _update_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            for exp in record["Exps"]:
                if exp["Exp ID"] == self.global_id:
                    exp.update(self._extra_record_data)
        with open(self.record_file_path, 'w') as f:
            json.dump(record, f, indent=2)


class Run(_EntityBase):
    entity_type = "run"

    def __init__(self, name: str, desc: str, proj_id: int, exp_id: int, job_type: str,
                 output_dir: str, resume_from: Optional[str] = None):
        self.exp_id = exp_id
        self.job_type = job_type
        self.is_resuming = False
        if job_type == "fit" and resume_from:
            self.is_resuming = True
            match_proj_name = re.search(r"[/\\](proj_\d+_.*?)[/\\]", resume_from)
            match_exp_name = re.search(r"[/\\](exp_\d+_.*?)[/\\]", resume_from)
            match_run_name = re.search(r"[/\\](run_\d+_.*?)[/\\]", resume_from)
            if match_proj_name and match_exp_name and match_run_name:
                self.proj_name = match_proj_name.group(1)
                self.exp_name = match_exp_name.group(1)
                run_name = match_run_name.group(1)
                self.proj_dir = osp.join(output_dir, self.proj_name)
                self.exp_dir = osp.join(self.proj_dir, self.exp_name)
                self.run_dir = osp.join(self.exp_dir, run_name)
                super(Run, self).__init__(relative_dir=self.proj_dir, name=name, desc=desc,
                                          record_file_path=osp.join(self.proj_dir, self.proj_name + ".json"),
                                          output_dir=output_dir)
                # Update attributes' values
                self.global_id = int(run_name.split('_')[1])
                self.name = run_name
                self._read_from_record_file()
                self._process_metrics_csv()
            else:
                warnings.warn(f"The original run id CAN NOT be induced from {resume_from}, "
                              f"a new run will be created.")
                self._create_new_run(name, desc, proj_id, exp_id, output_dir)
        else:
            self._create_new_run(name, desc, proj_id, exp_id, output_dir)

    def _create_new_run(self, name: str, desc: str, proj_id: int, exp_id: int, output_dir: str):
        self.proj_name, self.exp_name = get_name_from_id(output_dir, [proj_id, exp_id], id_type="exp")
        self.proj_dir = osp.join(output_dir, self.proj_name)
        self.exp_dir = osp.join(self.proj_dir, self.exp_name)
        super().__init__(relative_dir=self.proj_dir, name=name, desc=desc,
                         record_file_path=osp.join(self.proj_dir, self.proj_name + ".json"),
                         output_dir=output_dir)
        self.run_dir = osp.join(self.exp_dir, self.name)

        os.mkdir(self.run_dir)
        self._write_new_record_entry()
        self.print_message(self.run_dir)

    # ############# Dealing with pytorch_lightning.loggers.CSVLogger's metrics.csv
    def _process_metrics_csv(self):
        """
        Since pytorch_lightning.loggers.CSVLogger will override previous "metrics.csv",
        special process is needed when resuming fit to avoid "metrics.csv" being overridden.
        """
        metrics_csv_path = osp.join(self.run_dir, "metrics.csv")
        if osp.exists(metrics_csv_path):
            # Rename the original "metrics.csv" to "metrics_<max_cnt>.csv",
            # to reserve the filename "metrics.csv" for future fit resuming.
            max_cnt = get_next_id(self.run_dir, id_type='metrics_csv')
            shutil.move(metrics_csv_path,
                        metrics_csv_path[:-4] + f"_{max_cnt}.csv")

    @rank_zero_only
    def merge_metrics_csv(self):
        """
        Merge multiple metrics_csv (if any) into "merged_metrics.csv".
        """
        metrics_csv_files = [filename for filename in os.listdir(self.run_dir)
                             if filename.startswith("metrics_") and filename.endswith(".csv")]
        metrics_csv_files.sort(key=lambda x: int(osp.splitext(x)[0].split('_')[1]))
        if osp.exists(osp.join(self.run_dir, "metrics.csv")):
            metrics_csv_files.append("metrics.csv")
        if len(metrics_csv_files) > 1:
            with open(osp.join(self.run_dir, "merged_metrics.csv"), 'w') as merged_metrics_csv:
                for csv_idx, csv_filename in enumerate(metrics_csv_files):
                    with open(osp.join(self.run_dir, csv_filename)) as metrics_csv:
                        if csv_idx != 0:
                            next(metrics_csv)  # skip the csv header to avoid repetition
                        merged_metrics_csv.write(metrics_csv.read())

    # ######################################################

    def _get_record_dict(self) -> dict:
        return {
            "Run ID": self.global_id,
            "Run Name": self.name,
            "Run Description": self.desc,
            "Created Time": self.created_time,
            "Job Type": self.job_type
        }

    def _read_from_record_file(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
        for exp in record["Exps"]:
            if exp["Exp ID"] == self.exp_id:
                for run in exp["Runs"]:
                    if run["Run ID"] == self.global_id:
                        self.name = run["Run Name"]
                        self.desc = run["Run Description"]
                        self.created_time = run["Created Time"]
                        self.job_type = run["Job Type"]
                        self._extra_record_data.update(run)
                        for key in self._get_record_dict().keys():
                            del self._extra_record_data[key]

    def _write_new_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            for exp in record["Exps"]:
                if exp["Exp ID"] == self.exp_id:
                    exp["Runs"].append(self._get_record_entry())
        with open(self.record_file_path, 'w') as f:
            json.dump(record, f, indent=2)

    def _update_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            for exp in record["Exps"]:
                if exp["Exp ID"] == self.exp_id:
                    for run in exp["Runs"]:
                        if run["Run ID"] == self.global_id:
                            run.update(self._extra_record_data)
        with open(self.record_file_path, 'w') as f:
            json.dump(record, f, indent=2)
