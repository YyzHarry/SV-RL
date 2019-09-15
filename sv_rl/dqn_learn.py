import os
import sys
import time
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer
from utils.gym_utils import get_wrapper_by_name
from utils.svrl_utils import *
import logz

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])


class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


def dqn_learning(
        env,
        method,
        game,
        q_func,
        optimizer_spec,
        exploration,
        stopping_criterion=None,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=4,
        frame_history_len=4,
        target_update_freq=10000,
        double=False,
        dueling=False,
        logdir=None,
        svrl=False,
        me_type=None,
        maskp=None,
        maskstep=None,
        maskscheduler=True
    ):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space) == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    def select_epsilon_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            with torch.no_grad():
                return model(Variable(obs)).data.max(1)[1].view(1, 1)
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    #   RUN ENV   #
    ###############

    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    SAVE_MODEL_EVERY_N_STEPS = 1000000
    mask_scheduler_step = (1 - maskp) / maskstep

    for t in count():
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ################
        # STEP THE ENV #
        ################

        last_idx = replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()

        if t > learning_starts:
            action = select_epsilon_greedy_action(Q, recent_observations, t)[0][0]
        else:
            action = random.randrange(num_actions)

        obs, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0))
        replay_buffer.store_effect(last_idx, action, reward, done)

        if done:
            obs = env.reset()
        last_obs = obs

        ################
        #   TRAINING   #
        ################

        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):

            # mask scheduler
            if maskscheduler:
                maskp = min(maskp + mask_scheduler_step, 1)

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

            if USE_CUDA:
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()

            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            target_q_mat = target_Q(next_obs_batch).detach()

            # SV-RL scheme
            if svrl:
                target_q_mat = globals()[me_type](target_q_mat, target_q_mat.size(0), target_q_mat.size(1), maskp)

            if not double:
                next_max_q = target_q_mat.max(1)[0]
            else:
                q_temp = Q(next_obs_batch).detach()
                act_temp = np.argmax(q_temp.cpu(), axis=1)
                next_max_q = torch.sum(torch.from_numpy(np.eye(num_actions)[act_temp]).type(dtype) * target_q_mat.type(dtype), dim=1)

            next_Q_values = not_done_mask * next_max_q.type(dtype)
            target_Q_values = rew_batch + (gamma * next_Q_values)

            loss = F.smooth_l1_loss(current_Q_values, target_Q_values)

            optimizer.zero_grad()
            loss.backward()

            for params in Q.parameters():
                params.grad.data.clamp_(-1, 1)

            optimizer.step()
            num_param_updates += 1

            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        ################
        # LOG PROGRESS #
        ################

        # save model
        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            add_str = 'single'
            if double:
                add_str = 'double'
            if dueling:
                add_str = 'dueling'
            model_save_path = 'models/%s_%s_%s.ckpt' % (str(game[:-14]), add_str, method)
            torch.save(Q.state_dict(), model_save_path)

        # log process
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:

            logz.log_tabular('Timestep', t)
            logz.log_tabular('MeanReward100Episodes', mean_episode_reward)
            logz.log_tabular('BestMeanReward', best_mean_episode_reward)
            logz.log_tabular('Episodes', len(episode_rewards))
            logz.log_tabular('Exploration', exploration.value(t))
            logz.dump_tabular()

            sys.stdout.flush()
