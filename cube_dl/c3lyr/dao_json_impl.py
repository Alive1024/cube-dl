import json
import os
import os.path as osp
import re
import shutil
from collections import OrderedDict
from datetime import datetime

from jsonpath_ng import parse

from .dao import ExperimentDAO, ProjectDAO, RunDAO
from .entities import Experiment, Project, Run


# >>>>>>>>>>>>>>>>>>>> Functions for Reading & Writing JSON Files >>>>>>>>>>>>>>>>>>>>
def _json_read_from_path(json_path) -> dict:
    with open(json_path, encoding="utf-8") as f:
        record = json.loads(f.read())
    return record


def _json_dump_to_file(obj, json_path):
    with open(json_path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


# >>>>>>>>>>>>>> Functions for Getting the Path to the Project Record File >>>>>>>>>>>>>>
def _get_record_file_path_from_proj(proj: Project, ensure_existence=True):
    record_file_path = osp.join(proj.proj_dir, proj.dirname + ".json")
    if ensure_existence and (not osp.exists(record_file_path)):
        raise FileNotFoundError(f"There is no record file in {proj.proj_dir}.")
    return record_file_path


def _get_record_file_path_from_proj_id(proj_id: str, output_dir: str):
    record_file_path = None
    for name in os.listdir(output_dir):
        if name.startswith(f"{Project.ENTITY_TYPE}_{proj_id}_"):
            record_file_path = osp.join(output_dir, name, name + ".json")
    if record_file_path is None:
        raise FileNotFoundError(f'There is no record file with id "{proj_id}" in "{output_dir}".')
    return record_file_path


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class ProjectDAOJsonImpl(ProjectDAO):
    def set_extra_data(self, proj: Project, **kwargs):
        self.update_entry(proj, **kwargs)

    def get_extra_data(self, proj: Project, key: str):
        record = _json_read_from_path(_get_record_file_path_from_proj(proj))
        if key not in record:
            raise KeyError(f'There is no data with key "{key}".')
        return record[key]

    def insert_entry(self, proj: Project):
        entry = OrderedDict(
            {
                "ID": proj.global_id,
                "Name": proj.name,
                "Desc": proj.desc,
                "CreatedTime": proj.created_time,
                "Path": proj.proj_dir,
                "Exps": OrderedDict(),
            }
        )
        _json_dump_to_file(entry, _get_record_file_path_from_proj(proj, ensure_existence=False))

    def update_entry(self, proj: Project, **kwargs):
        record_file_path = _get_record_file_path_from_proj(proj)
        record = _json_read_from_path(record_file_path)
        for key, value in kwargs.items():
            parser = parse(f"$['{key}']")
            parser.update_or_create(record, value)
        _json_dump_to_file(record, record_file_path)

    @staticmethod
    def get_proj_from_id(output_dir: str, proj_id: str) -> Project:
        record = _json_read_from_path(_get_record_file_path_from_proj_id(proj_id, output_dir))
        proj = Project()
        proj.global_id = proj_id
        proj.name = record["Name"]
        proj.desc = record["Desc"]
        proj.created_time = record["CreatedTime"]
        proj.proj_dir = record["Path"]
        return proj

    @staticmethod
    def get_projects(output_dir: str) -> list[OrderedDict]:
        projects = []
        for name in os.listdir(output_dir):
            if osp.isdir(osp.join(output_dir, name)):
                match = re.match(f"{Project.ENTITY_TYPE}_(.+?)_.+", name)
                if match:
                    record = _json_read_from_path(osp.join(output_dir, name, name + ".json"))
                    projects.append(
                        OrderedDict(
                            {
                                "ID": record["ID"],
                                "Name": record["Name"],
                                "Desc": record["Desc"],
                                "CreatedTime": record["CreatedTime"],
                            }
                        )
                    )
        # Sort the list according to the created time
        projects.sort(key=lambda proj: datetime.strptime(proj["CreatedTime"], "%Y-%m-%d %H:%M:%S"))
        return projects

    @staticmethod
    def get_all_projects_exps(output_dir: str) -> list[OrderedDict]:
        projects_exps = ProjectDAOJsonImpl.get_projects(output_dir)
        for proj in projects_exps:
            proj["Exps"] = ExperimentDAOJsonImpl.get_exps_of(output_dir, proj["ID"])
        return projects_exps

    def remove_entry(self, proj: Project, **kwargs):
        shutil.rmtree(proj.proj_dir)


class ExperimentDAOJsonImpl(ExperimentDAO):
    def set_extra_data(self, exp: Experiment, **kwargs):
        self.update_entry(exp, **kwargs)

    def get_extra_data(self, exp: Experiment, key: str):
        record_exp = _json_read_from_path(_get_record_file_path_from_proj(exp.belonging_proj))["Exps"][exp.global_id]
        if key not in record_exp:
            raise KeyError(f'There is no data with key "{key}".')
        return record_exp[key]

    def insert_entry(self, exp: Experiment):
        entry = OrderedDict(
            {
                "Name": exp.name,
                "Desc": exp.desc,
                "CreatedTime": exp.created_time,
                "Path": exp.exp_dir,
                "Runs": OrderedDict(),
            }
        )
        record_file_path = _get_record_file_path_from_proj(exp.belonging_proj)
        record = _json_read_from_path(record_file_path)
        record["Exps"][exp.global_id] = entry  # use id as key
        _json_dump_to_file(record, record_file_path)

    def update_entry(self, exp: Experiment, **kwargs):
        record_file_path = _get_record_file_path_from_proj(exp.belonging_proj)
        record = _json_read_from_path(record_file_path)
        for key, value in kwargs.items():
            parser = parse(f"$['Exps']['{exp.global_id}']['{key}']")
            parser.update_or_create(record, value)
        _json_dump_to_file(record, record_file_path)

    @staticmethod
    def get_exp_from_id(output_dir: str, proj_id: str, exp_id: str) -> Experiment:
        record_exp = _json_read_from_path(_get_record_file_path_from_proj_id(proj_id, output_dir))["Exps"][exp_id]
        exp = Experiment()
        exp.global_id = exp_id
        exp.name = record_exp["Name"]
        exp.desc = record_exp["Desc"]
        exp.created_time = record_exp["CreatedTime"]
        exp.exp_dir = record_exp["Path"]
        exp.belonging_proj = ProjectDAOJsonImpl.get_proj_from_id(output_dir, proj_id)
        return exp

    @staticmethod
    def get_exps_of(output_dir: str, proj_id: str) -> list[OrderedDict]:
        exps = []
        record_exps = _json_read_from_path(_get_record_file_path_from_proj_id(proj_id, output_dir))["Exps"]
        for exp_id, exp in record_exps.items():
            exps.append(
                OrderedDict(
                    {
                        "ID": exp_id,
                        "Name": exp["Name"],
                        "Desc": exp["Desc"],
                        "CreatedTime": exp["CreatedTime"],
                    }
                )
            )
        return exps

    @staticmethod
    def get_all_exps_runs(output_dir: str, proj_id: str) -> list[OrderedDict]:
        exps_runs = ExperimentDAOJsonImpl.get_exps_of(output_dir, proj_id)
        for exp in exps_runs:
            exp["Runs"] = RunDAOJsonImpl.get_runs_of(output_dir, proj_id, exp["ID"])
        return exps_runs

    def remove_entry(self, exp: Experiment, **kwargs):
        shutil.rmtree(exp.exp_dir)
        record_file_path = _get_record_file_path_from_proj(exp.belonging_proj)
        record = _json_read_from_path(record_file_path)
        del record["Exps"][exp.global_id]
        _json_dump_to_file(record, record_file_path)


class RunDAOJsonImpl(RunDAO):
    def set_extra_data(self, run: Run, **kwargs):
        self.update_entry(run, **kwargs)

    def get_extra_data(self, run: Run, key: str):
        record = _json_read_from_path(_get_record_file_path_from_proj(run.belonging_exp.belonging_proj))
        record_run = record["Exps"][run.belonging_exp.global_id]["Runs"][run.global_id]
        if key not in record_run:
            raise KeyError(f'There is no data with key "{key}".')
        return record_run[key]

    def insert_entry(self, run: Run):
        entry = OrderedDict(
            {
                "Name": run.name,
                "Desc": run.desc,
                "CreatedTime": run.created_time,
                "Path": run.run_dir,
                "Type": run.job_type,
            }
        )
        record_file_path = _get_record_file_path_from_proj(run.belonging_exp.belonging_proj)
        record = _json_read_from_path(record_file_path)
        record["Exps"][run.belonging_exp.global_id]["Runs"][run.global_id] = entry  # use id as key
        _json_dump_to_file(record, record_file_path)

    def update_entry(self, run: Run, **kwargs):
        belonging_exp = run.belonging_exp
        record_file_path = _get_record_file_path_from_proj(belonging_exp.belonging_proj)
        record = _json_read_from_path(record_file_path)
        for key, value in kwargs.items():
            parser = parse(f"$['Exps']['{belonging_exp.global_id}']['Runs']['{run.global_id}']['{key}']")
            parser.update_or_create(record, value)
        _json_dump_to_file(record, record_file_path)

    @staticmethod
    def get_run_from_id(output_dir: str, proj_id: str, exp_id: str, run_id: str) -> Run:
        record = _json_read_from_path(_get_record_file_path_from_proj_id(proj_id, output_dir))
        record_run = record["Exps"][exp_id]["Runs"][run_id]
        run = Run()
        run.global_id = run_id
        run.name = record_run["Name"]
        run.desc = record_run["Desc"]
        run.created_time = record_run["CreatedTime"]
        run.run_dir = record_run["Path"]
        run.job_type = record_run["Type"]
        run.belonging_exp = ExperimentDAOJsonImpl.get_exp_from_id(output_dir, proj_id, exp_id)
        return run

    @staticmethod
    def parse_ids_from_ckpt_path(ckpt_path: str):
        match_proj_id = re.search(r"[/\\]proj_(.+?)_.*?[/\\]", ckpt_path)
        match_exp_id = re.search(r"[/\\]exp_(.+?)_.*?[/\\]", ckpt_path)
        match_run_id = re.search(r"[/\\]run_(.+?)_.*?[/\\]", ckpt_path)
        if match_proj_id and match_exp_id and match_run_id:
            return match_proj_id.group(1), match_exp_id.group(1), match_run_id.group(1)
        else:
            raise ValueError(f"The original run id CAN NOT be induced from {ckpt_path}")

    @staticmethod
    def get_runs_of(output_dir: str, proj_id: str, exp_id: str) -> list[OrderedDict]:
        runs = []
        record = _json_read_from_path(_get_record_file_path_from_proj_id(proj_id, output_dir))
        record_runs = record["Exps"][exp_id]["Runs"]
        for run_id, run in record_runs.items():
            runs.append(
                OrderedDict(
                    {
                        "ID": run_id,
                        "Name": run["Name"],
                        "Desc": run["Desc"],
                        "CreatedTime": run["CreatedTime"],
                    }
                )
            )
        return runs

    def remove_entry(self, run: Run, **kwargs):
        shutil.rmtree(run.run_dir)
        record_file_path = _get_record_file_path_from_proj(run.belonging_exp.belonging_proj)
        record = _json_read_from_path(record_file_path)
        del record["Exps"][run.belonging_exp.global_id]["Runs"][run.global_id]
        _json_dump_to_file(record, record_file_path)
