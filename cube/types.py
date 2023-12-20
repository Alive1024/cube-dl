from typing import Literal

# Job types of a run
JOB_TYPES_T = Literal["fit", "resume-fit", "validate", "test", "predict", "tune"]

# Types of runner
RUNNER_TYPES_T = Literal["default", "fit", "validate", "test", "predict", "tune"]
