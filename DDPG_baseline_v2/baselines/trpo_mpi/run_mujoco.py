#!/usr/bin/env python
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds, tf_util as U
import os.path as osp
import gym
import logging
import argparse
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
import tensorflow as tf
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_mpi

import sys

def train(env_id, num_timesteps, seed, policy_hid_size, vf_hid_size, activation_policy, activation_vf):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            policy_hid_size=policy_hid_size, vf_hid_size=vf_hid_size, activation_policy=activation_policy, activation_vf=activation_vf)
    env = bench.Monitor(env, logger.get_dir() and 
        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=int(5e3), max_kl=0.01, cg_iters=20, cg_damping=0.1,
        max_timesteps=int(num_timesteps), gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    parser = argparse.ArgumentParser(description='TRPO.')
    parser.add_argument("task", type=str)
    parser.add_argument("--policy_size", nargs="+", default=(64,64), type=int)
    parser.add_argument("--value_func_size", nargs="+", default=(64,64), type=int)
    parser.add_argument("--activation_vf", type=str, default="tanh")
    parser.add_argument("--activation_policy", type=str, default="tanh")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_dir", type=str, default="./logs/")
    activation_map = { "relu" : tf.nn.relu, "leaky_relu" : U.lrelu, "tanh" :tf.nn.tanh}

    args = parser.parse_args()
    logger.configure(dir=args.log_dir)
    activation_policy = activation_map[args.activation_policy]
    activation_vf = activation_map[args.activation_vf]

    train(args.task, 2e6, args.seed, args.policy_size, args.value_func_size, activation_policy,  activation_vf)


if __name__ == '__main__':
    main()
