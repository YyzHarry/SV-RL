import gym
from gym import wrappers

from .seed import set_global_seeds
from .atari_wrapper import wrap_deepmind, wrap_deepmind_ram


def get_env(task, seed):
    env = task

    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = 'tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind(env)

    return env


def get_ram_env(env, seed):
    set_global_seeds(seed)
    env.seed(seed)

    expt_dir = '/tmp/gym-results'
    env = wrappers.Monitor(env, expt_dir, force=True)
    env = wrap_deepmind_ram(env)

    return env


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s" % classname)
