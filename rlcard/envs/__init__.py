''' Register new environments
'''
from rlcard.envs.env import Env
from rlcard.envs.registration import register, make

register(
    env_id='limit-holdem',
    entry_point='rlcard.envs.limitholdem:LimitholdemEnv',
)

register(
    env_id='no-limit-holdem',
    entry_point='rlcard.envs.nolimitholdem:NolimitholdemEnv',
)

register(
    env_id='multi-leduc-holdem',
    entry_point='rlcard.envs.multileducholdem:MultiLeducholdemEnv'
)
