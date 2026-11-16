"""Final MuJoCo environment and hybrid reinforcement-learning controller."""
from .environment import MuscleHumanoid
from .rl_env import ResidualHumanoid

__all__ = ["MuscleHumanoid", "ResidualHumanoid"]
