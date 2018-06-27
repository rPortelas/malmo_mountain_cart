import distutils.version
import os
import sys
import warnings

from gym2 import error
from gym2.utils import reraise
from gym2.version import VERSION as __version__

from gym2.core import Env, GoalEnv, Space, Wrapper, ObservationWrapper, ActionWrapper, RewardWrapper
from gym2.envs import make, spec
from gym2 import wrappers, spaces, logger

def undo_logger_setup():
    warnings.warn("gym2.undo_logger_setup is deprecated. gym2 no longer modifies the global logging configuration")

__all__ = ["Env", "Space", "Wrapper", "make", "spec", "wrappers"]
