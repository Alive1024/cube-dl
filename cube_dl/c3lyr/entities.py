import secrets
import string
from abc import ABCMeta
from typing import Literal


def generate_id(length: int = 8) -> str:
    """
    Generate a random base-36 string of `length` digits.
    Borrowed from `wandb.sdk.lib.runid.generate_id`.
    """
    # There are ~2.8T base-36 8-digit strings. If we generate 210k ids,
    # we'll have a ~1% chance of collision.
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class _EntityBase(metaclass=ABCMeta):
    ENTITY_TYPE: Literal["proj", "exp", "run"] = ""

    def __init__(self):
        self._global_id = None
        self._name = None
        self._desc = None
        self._dirname = None
        self._created_time = None

    @property
    def global_id(self):
        return self._global_id

    @global_id.setter
    def global_id(self, value: str):
        self._global_id = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        # Update dirname after name is assigned
        self._dirname = f"{self.ENTITY_TYPE}_{self.global_id}_{self.name}"

    @property
    def desc(self):
        return self._desc

    @desc.setter
    def desc(self, value: str):
        self._desc = value

    @property
    def dirname(self):
        return self._dirname

    @property
    def created_time(self):
        return self._created_time

    @created_time.setter
    def created_time(self, value: str):
        self._created_time = value


class Project(_EntityBase):
    ENTITY_TYPE = "proj"

    def __init__(self):
        super().__init__()
        self._proj_dir = None
        self._logger = None

    @property
    def proj_dir(self):
        return self._proj_dir

    @proj_dir.setter
    def proj_dir(self, value: str):
        self._proj_dir = value


class Experiment(_EntityBase):
    ENTITY_TYPE = "exp"

    def __init__(self):
        super().__init__()
        self._belonging_proj = None
        self._exp_dir = None

    @property
    def belonging_proj(self) -> Project:
        return self._belonging_proj

    @belonging_proj.setter
    def belonging_proj(self, value: Project):
        self._belonging_proj = value

    @property
    def exp_dir(self):
        return self._exp_dir

    @exp_dir.setter
    def exp_dir(self, value: str):
        self._exp_dir = value


class Run(_EntityBase):
    ENTITY_TYPE = "run"

    def __init__(self):
        super().__init__()
        self._belonging_exp = None
        self._run_dir = None
        self._job_type = None
        self._is_resuming = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value
        # Update dirname after name is assigned
        # Insert the job type for `Run`
        self._dirname = f"{self.ENTITY_TYPE}_{self.global_id}_{self._job_type}_{self.name}"

    @property
    def belonging_exp(self) -> Experiment:
        return self._belonging_exp

    @belonging_exp.setter
    def belonging_exp(self, value: Experiment):
        self._belonging_exp = value

    @property
    def run_dir(self):
        return self._run_dir

    @run_dir.setter
    def run_dir(self, value: str):
        self._run_dir = value

    @property
    def job_type(self):
        return self._job_type

    @job_type.setter
    def job_type(self, value):
        self._job_type = value

    @property
    def is_resuming(self):
        return self._is_resuming

    @is_resuming.setter
    def is_resuming(self, value):
        self._is_resuming = value


ENTITY_T = Project | Experiment | Run
