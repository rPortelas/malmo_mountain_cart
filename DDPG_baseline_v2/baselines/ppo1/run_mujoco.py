#!/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
import argparse
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
import sys
import tensorflow as tf

def train(env_id, num_timesteps, seed, policy_hid_size, vf_hid_size, activation_policy, activation_vf):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            policy_hid_size=policy_hid_size, vf_hid_size=vf_hid_size, activation_policy=activation_policy, activation_vf=activation_vf)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.995, lam=0.97, schedule='constant',
        )
    env.close()

def main():
    parser = argparse.ArgumentParser(description='PPO.')
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
