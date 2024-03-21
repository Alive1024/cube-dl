"""`c3lyr` stands for "Core Triple Layers".

This package implements the triple layer concepts: project, experiment and run.
"""

import os
import os.path as osp
from datetime import datetime

from .dao import ExperimentDAO, ProjectDAO, RunDAO
from .dao_json_impl import ExperimentDAOJsonImpl, ProjectDAOJsonImpl, RunDAOJsonImpl
from .entities import Experiment, Project, Run, generate_id
from .run_context import dump_run, load_run, remove_run

__all__ = [
    "Project",
    "Experiment",
    "Run",
    "EntityFactory",
    "DAOFactory",
    "remove_run",
    "load_run",
    "dump_run",
]


def _make_dir(target_dir, created_type: str, print_message=True):
    if not osp.exists(target_dir):
        os.mkdir(target_dir)
    if print_message:
        print(f'{created_type}: "{osp.split(target_dir)[1]}" created, storage path: {target_dir}')


class EntityFactory:
    """
    Exported factory class for producing new entity objects.
    """

    @staticmethod
    def _set_common(entity, name: str, desc: str, global_id: str):
        entity.global_id = global_id
        entity.name = name
        entity.desc = desc
        entity.created_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def get_proj_instance(name: str, desc: str, output_dir: str) -> Project:
        proj = Project()
        EntityFactory._set_common(proj, name, desc, global_id=generate_id(2))
        proj.proj_dir = osp.join(output_dir, proj.dirname)
        _make_dir(proj.proj_dir, proj.ENTITY_TYPE)
        return proj

    @staticmethod
    def get_exp_instance(name: str, desc: str, output_dir: str, proj_id: str) -> Experiment:
        exp = Experiment()
        EntityFactory._set_common(exp, name, desc, global_id=generate_id(2))
        exp.belonging_proj = DAOFactory.get_proj_dao().get_proj_from_id(output_dir, proj_id)
        exp.exp_dir = osp.join(exp.belonging_proj.proj_dir, exp.dirname)
        _make_dir(exp.exp_dir, exp.ENTITY_TYPE)
        # Create an empty ".gitkeep" within the exp directory
        with open(osp.join(exp.exp_dir, ".gitkeep"), "w") as f:
            f.write("")
        return exp

    @staticmethod
    def get_run_instance(name: str, desc: str, output_dir: str, proj_id: str, exp_id: str, job_type: str) -> Run:
        run = Run()
        run.job_type = job_type  # set `job_type` before setting `name` (as `dirname` needs `job_type`)
        EntityFactory._set_common(run, name, desc, global_id=generate_id(4))
        run.belonging_exp = DAOFactory.get_exp_dao().get_exp_from_id(output_dir, proj_id, exp_id)
        run.run_dir = osp.join(run.belonging_exp.exp_dir, run.dirname)
        _make_dir(run.run_dir, run.ENTITY_TYPE)
        return run


class DAOFactory:
    """
    Exported factory class for producing DAO objects.
    These DAO objects' details depend on concrete implementation, JSON implementation is used for now.
    """

    @staticmethod
    def get_proj_dao() -> ProjectDAO:
        return ProjectDAOJsonImpl()

    @staticmethod
    def get_exp_dao() -> ExperimentDAO:
        return ExperimentDAOJsonImpl()

    @staticmethod
    def get_run_dao() -> RunDAO:
        return RunDAOJsonImpl()
