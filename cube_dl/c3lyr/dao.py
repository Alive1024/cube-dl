from abc import ABCMeta, abstractmethod
from collections import OrderedDict

from .entities import ENTITY_T, Project


class _EntityDAOBase(metaclass=ABCMeta):
    # >>>>>>>>>>>> Methods for Setting and Getting Custom Data >>>>>>>>>>>>
    @abstractmethod
    def set_extra_data(self, entity: ENTITY_T, **kwargs):
        pass

    @abstractmethod
    def get_extra_data(self, entity: ENTITY_T, key: str):
        pass

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    @abstractmethod
    def insert_entry(self, entity: ENTITY_T):
        pass

    @abstractmethod
    def update_entry(self, entity: ENTITY_T, **kwargs):
        pass

    @abstractmethod
    def remove_entry(self, entity: ENTITY_T, **kwargs):
        pass


class ProjectDAO(_EntityDAOBase, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_proj_from_id(output_dir: str, proj_id: str) -> Project:
        pass

    @staticmethod
    @abstractmethod
    def get_projects(output_dir: str) -> list[OrderedDict]:
        pass

    @staticmethod
    @abstractmethod
    def get_all_projects_exps(output_dir: str) -> list[OrderedDict]:
        pass


class ExperimentDAO(_EntityDAOBase, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_exp_from_id(output_dir: str, proj_id: str, exp_id: str):
        pass

    @staticmethod
    @abstractmethod
    def get_exps_of(output_dir: str, proj_id: str) -> list[OrderedDict]:
        pass

    @staticmethod
    @abstractmethod
    def get_all_exps_runs(output_dir: str, proj_id: str) -> list[OrderedDict]:
        pass


class RunDAO(_EntityDAOBase, metaclass=ABCMeta):
    @staticmethod
    @abstractmethod
    def get_run_from_id(output_dir: str, proj_id: str, exp_id: str, run_id: str):
        pass

    @staticmethod
    @abstractmethod
    def parse_ids_from_ckpt_path(ckpt_path: str):
        pass

    @staticmethod
    @abstractmethod
    def get_runs_of(output_dir: str, proj_id: str, exp_id: str) -> list[OrderedDict]:
        pass
