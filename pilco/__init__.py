from . import models
from . import controllers
from . import rewards
from . import envs

from gym.envs.registration import register
register(
    id='acrobot-continuous-v1',
    entry_point='pilco.envs:ContinuousAcrobotEnv'
)