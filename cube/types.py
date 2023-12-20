from collections.abc import Callable
from typing import Literal

from cube.c3lyr import Run

# Job types of a run
JOB_TYPES_T = Literal["fit", "resume-fit", "validate", "test", "predict", "tune"]

# Types of runner
RUNNER_TYPES_T = Literal["default", "fit", "validate", "test", "predict", "tune"]


LOGGER_GETTER_T = Callable[[Run], Callable | bool]
LOGGER_T = Callable | bool
