import os
import gym
from gym import wrappers
import random
import torch.optim as optim

from dqn_model import DQN, Dueling_DQN
from dqn_learn import OptimizerSpec, dqn_learning
from utils.schedule import LinearSchedule
from utils.gym_utils import get_env, get_wrapper_by_name
import logz


REPLAY_BUFFER_SIZE = 100000
LEARNING_STARTS = 10000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00001
# ALPHA = 0.95
# EPS = 0.01


def atari_learn(env, args, num_timesteps):
    logdir = os.path.join('data', args.exp_name)

    num_iterations = float(num_timesteps) / 4.0

    # lr_multiplier = 1.0
    # lr_schedule = PiecewiseSchedule([
    #     (0, 1e-4 * lr_multiplier),
    #     (num_iterations / 10, 1e-4 * lr_multiplier),
    #     (num_iterations / 2, 5e-5 * lr_multiplier),
    # ],
    #     outside_value=5e-5 * lr_multiplier)
    # optimizer = dqn.OptimizerSpec(
    #     constructor=tf.train.AdamOptimizer,
    #     kwargs=dict(epsilon=1e-4),
    #     lr_schedule=lr_schedule
    # )

    def stopping_criterion(env):
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # optimizer_spec = OptimizerSpec(
    #     constructor=optim.RMSprop,
    #     kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    # )

    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE),
    )

    exploration_schedule = LinearSchedule(30000, 0.01)

    # exploration_schedule = PiecewiseSchedule(
    #     [
    #         (0, 1.0),
    #         (1e6, 0.1),
    #         (num_iterations / 2, 0.01),
    #     ], outside_value=0.01
    # )

    logz.configure_output_dir(logdir)

    if args.dueling:
        dqn_learning(
            env=env,
            method=args.method,
            game=args.env,
            q_func=Dueling_DQN,
            optimizer_spec=optimizer_spec,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double=args.double,
            dueling=args.dueling,
            logdir=logdir,
            svrl=args.svrl,
            me_type=args.me_type,
            maskp=args.maskp,
            maskstep=args.maskstep,
            maskscheduler=args.maskscheduler,
        )
    else:
        dqn_learning(
            env=env,
            method=args.method,
            game=args.env,
            q_func=DQN,
            optimizer_spec=optimizer_spec,
            exploration=exploration_schedule,
            stopping_criterion=stopping_criterion,
            replay_buffer_size=REPLAY_BUFFER_SIZE,
            batch_size=args.batch_size,
            gamma=args.gamma,
            learning_starts=LEARNING_STARTS,
            learning_freq=LEARNING_FREQ,
            frame_history_len=FRAME_HISTORY_LEN,
            target_update_freq=TARGET_UPDATE_FREQ,
            double=args.double,
            dueling=args.dueling,
            logdir=logdir,
            svrl=args.svrl,
            me_type=args.me_type,
            maskp=args.maskp,
            maskstep=args.maskstep,
            maskscheduler=args.maskscheduler,
        )

    env.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4', help='choose atari games')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount reward coefficient (default 0.99)')
    parser.add_argument('--batch-size', type=int, default=32, help='default: 32')
    parser.add_argument('--double', action='store_true', help='use double DQN')
    parser.add_argument('--dueling', action='store_true', help='use dueling DQN')
    parser.add_argument('--svrl', action='store_true')
    parser.add_argument('--maskscheduler', action='store_true')
    parser.add_argument('--method', type=str, default='baseline', help='name of the method')
    parser.add_argument('--me_type', type=str, default='softimp', help='choose the SV-RL mechanism')
    parser.add_argument('--maskp', type=float, default=0.9, help='mask probability (default: 0.9)')
    parser.add_argument('--maskstep', type=float, default=2e6, help='total steps for linear scheduler (default: 2e6)')
    parser.add_argument('--num_timesteps', type=float, default=6e7, help='total time steps (default: 6e7)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not (os.path.exists('data')):
        os.makedirs('data')

    # get Atari games
    task = gym.make(args.env)

    # run training
    seed = random.randint(0, 9999)
    print('random seed = %d' % seed)
    env = get_env(task, seed)
    atari_learn(env, args, num_timesteps=args.num_timesteps)


if __name__ == "__main__":
    main()
