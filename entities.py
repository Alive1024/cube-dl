"""
Entity classes implementing the "three-layer concepts": project, experiment and run.
Dealing with directories, filenames
"""
import os
import os.path as osp
from datetime import datetime
from abc import abstractmethod
from typing import Literal, Optional, OrderedDict
import re
import warnings
import shutil
import json

from pytorch_lightning.utilities.rank_zero import rank_zero_only


class _EntityBase:
    ENTITY_TYPE: Literal["proj", "exp", "run"]

    def __init__(self, relative_dir: str, name: str, desc: str, record_file_path: str, output_dir: str):
        self.global_id: int = self.get_next_id(relative_dir)
        self.name = f"{self.ENTITY_TYPE}_{self.global_id}_{name}"
        self.desc = desc
        self.record_file_path = record_file_path
        self.output_dir = output_dir
        self.created_time = datetime.now().strftime("%Y-%m-%d (%a) %H:%M:%S")

        self._extra_record_data = {}

    @rank_zero_only
    def _create_dir(self, target_dir, print_message=True):
        os.mkdir(target_dir)
        if print_message:
            print(f"{self.ENTITY_TYPE}: {self.name} created, storage path: {target_dir}")

    @staticmethod
    def _json_dump_to_file(obj, fp):
        json.dump(obj, fp, indent=2)

    # =========== Setter and getter for extra record data, which allows for save custom data. ===========
    def set_extra_record_data(self, **kwargs):
        self._extra_record_data.update(kwargs)
        self._update_record_entry()

    def get_extra_record_data(self, key: str):
        return self._extra_record_data[key]
    # ====================================================================================================

    @staticmethod
    @abstractmethod
    def get_next_id(target_dir) -> int:
        raise NotImplementedError

    @staticmethod
    def get_proj_name_from_id(output_dir, proj_id):
        proj_name = ""
        for fn in os.listdir(output_dir):
            if fn.startswith(f"proj_{proj_id}"):
                proj_name = fn
        if not proj_name:
            raise RuntimeError(f"Project with id {proj_id} NOT FOUND in {output_dir}.")
        return proj_name

    @staticmethod
    def get_proj_exp_names_from_ids(output_dir, proj_id, exp_id):
        proj_name, exp_name = "", ""
        for fn1 in os.listdir(output_dir):
            if fn1.startswith(f"proj_{proj_id}"):
                proj_name = fn1
                for fn2 in os.listdir(osp.join(output_dir, fn1)):
                    if fn2.startswith(f"exp_{exp_id}"):
                        exp_name = fn2
        if not proj_name:
            raise RuntimeError(f"Project with id {proj_id} NOT FOUND in {output_dir}.")
        if not exp_name:
            raise RuntimeError(f"Experiment with id {exp_id} NOT FOUND in {osp.join(output_dir, proj_name)}.")
        return proj_name, exp_name

    @abstractmethod
    def _get_record_dict(self) -> dict:
        raise NotImplementedError

    def _get_record_entry(self) -> dict:
        record_dict = self._get_record_dict()
        record_dict.update(self._extra_record_data)
        return record_dict

    @abstractmethod
    @rank_zero_only
    def _write_new_record_entry(self):
        raise NotImplementedError

    @abstractmethod
    @rank_zero_only
    def _update_record_entry(self):
        """
        Used when a key-value pair is set, i.e. when `set_extra_record_data` is called.
        """
        raise NotImplementedError


class Project(_EntityBase):
    ENTITY_TYPE = "proj"

    def __init__(self, name: str, desc: str, output_dir: str):
        super().__init__(relative_dir=output_dir, name=name, desc=desc,
                         record_file_path="", output_dir=output_dir)
        self.proj_dir = osp.join(output_dir, self.name)
        self._create_dir(self.proj_dir)
        self.record_file_path = osp.join(self.proj_dir, self.name + ".json")
        self._write_new_record_entry()

    @staticmethod
    def get_next_id(output_dir) -> int:
        indices = []
        for i in os.listdir(output_dir):
            if i.startswith(Project.ENTITY_TYPE + '_'):
                indices.append(int(i.split('_')[1]))
        return 1 if len(indices) == 0 else (max(indices) + 1)

    def _get_record_dict(self) -> dict:
        return {
            "Project ID": self.global_id,
            "Project Name": self.name,
            "Project Description": self.desc,
            "Created Time": self.created_time,
            "Storage Path": osp.abspath(self.output_dir),
            "Exps": []
        }

    @rank_zero_only
    def _write_new_record_entry(self):
        with open(self.record_file_path, 'w') as f:
            Project._json_dump_to_file(self._get_record_entry(), f)

    @rank_zero_only
    def _update_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            record.update(self._extra_record_data)
        with open(self.record_file_path, 'w') as f:
            Project._json_dump_to_file(record, f)


class Experiment(_EntityBase):
    ENTITY_TYPE = "exp"
    ARCHIVED_CONFIGS_DIRNAME = "archived_configs"

    def __init__(self, name: str, desc: str, proj_id: int, output_dir: str):
        proj_name = Experiment.get_proj_name_from_id(output_dir, proj_id)
        self.proj_dir = osp.join(output_dir, proj_name)

        super().__init__(relative_dir=self.proj_dir, name=name, desc=desc,
                         record_file_path=osp.join(self.proj_dir, proj_name + ".json"),
                         output_dir=output_dir)

        self.exp_dir = osp.join(self.proj_dir, self.name)
        self._create_dir(self.exp_dir)
        # Create the directory for archived configs
        self._create_dir(osp.join(self.exp_dir, Experiment.ARCHIVED_CONFIGS_DIRNAME), print_message=False)
        self._write_new_record_entry()

    @staticmethod
    def get_next_id(proj_dir) -> int:
        indices = []
        for i in os.listdir(proj_dir):
            if i.startswith(Experiment.ENTITY_TYPE + '_'):
                indices.append(int(i.split('_')[1]))
        return 1 if len(indices) == 0 else (max(indices) + 1)

    @staticmethod
    def get_archived_configs_dir(proj_id: int, exp_id: int, output_dir: str):
        proj_name, exp_name = Experiment.get_proj_exp_names_from_ids(output_dir, proj_id, exp_id)
        return osp.join(output_dir, proj_name, exp_name, Experiment.ARCHIVED_CONFIGS_DIRNAME)

    def _get_record_dict(self) -> dict:
        return {
            "Exp ID": self.global_id,
            "Exp Name": self.name,
            "Exp Description": self.desc,
            "Created Time": self.created_time,
            "Runs": []
        }

    @rank_zero_only
    def _write_new_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            record["Exps"].append(self._get_record_entry())
        with open(self.record_file_path, 'w') as f:
            Experiment._json_dump_to_file(record, f)

    @rank_zero_only
    def _update_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            for exp in record["Exps"]:
                if exp["Exp ID"] == self.global_id:
                    exp.update(self._extra_record_data)
        with open(self.record_file_path, 'w') as f:
            Experiment._json_dump_to_file(record, f)


class Run(_EntityBase):
    ENTITY_TYPE = "run"

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
                Run._process_metrics_csv(self.run_dir)
            else:
                warnings.warn(f"The original run id CAN NOT be induced from {resume_from}, "
                              f"a new run will be created.")
                self._create_new_run(name, desc, proj_id, exp_id, output_dir)
        else:
            self._create_new_run(name, desc, proj_id, exp_id, output_dir)

    def _create_new_run(self, name: str, desc: str, proj_id: int, exp_id: int, output_dir: str):
        self.proj_name, self.exp_name = Run.get_proj_exp_names_from_ids(output_dir, proj_id, exp_id)
        self.proj_dir = osp.join(output_dir, self.proj_name)
        self.exp_dir = osp.join(self.proj_dir, self.exp_name)
        super().__init__(relative_dir=self.proj_dir, name=name, desc=desc,
                         record_file_path=osp.join(self.proj_dir, self.proj_name + ".json"),
                         output_dir=output_dir)
        self.run_dir = osp.join(self.exp_dir, self.name)
        self._create_dir(self.run_dir)
        self._write_new_record_entry()

    @staticmethod
    def get_next_id(proj_dir) -> int:
        indices = []
        for i in os.listdir(proj_dir):
            if i.startswith("exp_") and osp.isdir(osp.join(proj_dir, i)):
                for j in os.listdir(osp.join(proj_dir, i)):
                    if j.startswith("run_"):
                        indices.append(int(j.split('_')[1]))
        return 1 if len(indices) == 0 else (max(indices) + 1)

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

    @rank_zero_only
    def _write_new_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            for exp in record["Exps"]:
                if exp["Exp ID"] == self.exp_id:
                    exp["Runs"].append(self._get_record_entry())
        with open(self.record_file_path, 'w') as f:
            Run._json_dump_to_file(record, f)

    @rank_zero_only
    def _update_record_entry(self):
        with open(self.record_file_path, 'r') as f:
            record = json.load(f)
            for exp in record["Exps"]:
                if exp["Exp ID"] == self.exp_id:
                    for run in exp["Runs"]:
                        if run["Run ID"] == self.global_id:
                            run.update(self._extra_record_data)
        with open(self.record_file_path, 'w') as f:
            Run._json_dump_to_file(record, f)

    # ================== Dealing with pytorch_lightning.loggers.CSVLogger's metrics.csv ==================
    @staticmethod
    @rank_zero_only
    def _process_metrics_csv(run_dir):
        """
        Since pytorch_lightning.loggers.CSVLogger will override previous "metrics.csv",
        special process is needed when resuming fit to avoid "metrics.csv" being overridden.
        """
        metrics_csv_path = osp.join(run_dir, "metrics.csv")
        if osp.exists(metrics_csv_path):
            # Rename the original "metrics.csv" to "metrics_<max_cnt>.csv",
            # to reserve the filename "metrics.csv" for future fit resuming.
            indices = []
            for filename in os.listdir(run_dir):
                match = re.match(r"metrics_(\d+).csv", filename)
                if match:
                    indices.append(int(match.group(1)))
            max_cnt = 1 if len(indices) == 0 else max(indices)
            shutil.move(metrics_csv_path,
                        metrics_csv_path[:-4] + f"_{max_cnt}.csv")

    @staticmethod
    @rank_zero_only
    def merge_metrics_csv(run_dir):
        """
        Merge multiple metrics_csv (if any) into "merged_metrics.csv".
        """
        metrics_csv_files = [filename for filename in os.listdir(run_dir)
                             if filename.startswith("metrics_") and filename.endswith(".csv")]
        metrics_csv_files.sort(key=lambda x: int(osp.splitext(x)[0].split('_')[1]))
        if osp.exists(osp.join(run_dir, "metrics.csv")):
            metrics_csv_files.append("metrics.csv")
        if len(metrics_csv_files) > 1:
            with open(osp.join(run_dir, "merged_metrics.csv"), 'w') as merged_metrics_csv:
                for csv_idx, csv_filename in enumerate(metrics_csv_files):
                    with open(osp.join(run_dir, csv_filename)) as metrics_csv:
                        if csv_idx != 0:
                            next(metrics_csv)  # skip the csv header to avoid repetition
                        merged_metrics_csv.write(metrics_csv.read())
    # =====================================================================================================

    # ================================== Exported Methods for Saving Files. ==================================
    @staticmethod
    @rank_zero_only
    def save_hparams(run_dir, hparams: OrderedDict):
        """
        Save hparams to json files. "hparams.json" always indicates the latest, similar to "metrics.csv".
        """
        hparams_json_path = osp.join(run_dir, "hparams.json")
        if osp.exists(hparams_json_path):
            indices = []
            for filename in os.listdir(run_dir):
                match = re.match(r"hparams_(\d+).json", filename)
                if match:
                    indices.append(int(match.group(1)))

            max_cnt = 1 if len(indices) == 0 else max(indices) + 1
            shutil.move(hparams_json_path, hparams_json_path[:-4] + f"_{max_cnt}.json")

        with open(hparams_json_path, 'w') as f:
            Run._json_dump_to_file(hparams, f)

    @staticmethod
    @rank_zero_only
    def remove_empty_hparams_yaml(run_dir):
        # Remove the empty "hparams.yaml" generated by PyTorch-Lightning
        hparams_yaml_path = osp.join(run_dir, "hparams.yaml")
        needs_removed = False
        if osp.exists(hparams_yaml_path):
            with open(hparams_yaml_path) as f:
                if f.read().strip() == "{}":
                    needs_removed = True
        if needs_removed:
            os.remove(hparams_yaml_path)

    @staticmethod
    @rank_zero_only
    def save_model_structure(run_dir, model):
        with open(osp.join(run_dir, "model_structure.txt"), 'w') as f:
            f.write(repr(model))
    # =========================================================================================================
