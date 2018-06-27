import os
import sys
#os.environ['LD_LIBRARY_PATH']+=':'+os.environ['HOME']+'/.mujoco/mjpro150/bin:'

sys.path.append('../../../')
sys.path.append('../../../../')

# sys.path.append('/home/flowers/Desktop/Scratch/DDPG_baseline_v2/')
import click
import numpy as np
import json
from mpi4py import MPI

from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.mpi_moments import mpi_moments
import baselines.her.experiment.config as config
from baselines.her.rollout import RolloutWorker
from baselines.her.util import mpi_fork
from baselines.her.replay_buffer import load_from_tulip
import pickle


def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]


def train(policy, rollout_worker, evaluator, env_name, num_cpu,
          n_epochs, n_test_rollouts, n_cycles, n_batches, policy_save_interval,
          save_policies, rank, gep_memory=None, buffer_location=None, study='HER', active_goal=False,  **kwargs):

    latest_policy_path = os.path.join(logger.get_dir(), 'policy_latest.pkl')
    best_policy_path = os.path.join(logger.get_dir(), 'policy_best.pkl')
    periodic_policy_path = os.path.join(logger.get_dir(), 'policy_{}.pkl')

    first_epoch = 0
    if 'GEP' in study:
        if gep_memory is not None:
            if type(gep_memory) is str:
                with open(gep_memory, 'rb') as f:
                    gep_mem = pickle.load(f)
            else:
                gep_mem=gep_memory
            buffer_gep = load_from_tulip(env_name, gep_mem)
            del gep_mem
            logger.info('Loading buffer from gep memory')
        else:
            assert buffer_location != '', 'study is ' + study + ', a buffer location should be provided.'
            # fill replay buffer
            logger.info('Cannot load buffer from file, NOT IMPLEMENTED')
        policy.store_episode(buffer_gep)
        size_buff = buffer_gep['o'].shape[0]
        logger.info('Buffer of ' + str(size_buff) + ' episodes has been loaded')
        first_epoch = int(size_buff / (2 * num_cpu * 50))

    #Book keeping
    episodes = []
    logger.info("Training...")
    best_success_rate = -1
    for epoch in range(first_epoch, n_epochs):
        # train
        rollout_worker.clear_history()
        for c in range(n_cycles):
            episode = rollout_worker.generate_rollouts()
            episodes.append(episode)
            policy.store_episode(episode)
            for _ in range(n_batches):
                policy.train()
            policy.update_target_net()

        # test
        evaluator.clear_history()
        for _ in range(n_test_rollouts):
            evaluator.generate_rollouts()

        # record logs
        logger.record_tabular('epoch', epoch)
        for key, val in evaluator.logs('test'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in rollout_worker.logs('train'):
            logger.record_tabular(key, mpi_average(val))
        for key, val in policy.logs():
            logger.record_tabular(key, mpi_average(val))

        if rank == 0:
            logger.dump_tabular()
        policy_path = periodic_policy_path.format(epoch)
        # save the policy if it's better than the previous ones
        success_rate = mpi_average(evaluator.current_success_rate())
        if rank == 0 and success_rate >= best_success_rate and save_policies:
            best_success_rate = success_rate
            logger.info('New best success rate: {}. Saving policy to {} ...'.format(best_success_rate, best_policy_path))
            evaluator.save_policy(best_policy_path)
            evaluator.save_policy(latest_policy_path)
        if rank == 0 and policy_save_interval > 0 and epoch % policy_save_interval == 0 and save_policies:
            policy_path = periodic_policy_path.format(epoch)
            logger.info('Saving all episodes info ...')
            pickle.dump(episodes, open( "./save/her_mmc.pickle", "wb" ))
            logger.info('Saving periodic policy to {} ...'.format(policy_path))
            evaluator.save_policy(policy_path)

        # make sure that different threads have different seeds
        local_uniform = np.random.uniform(size=(1,))
        root_uniform = local_uniform.copy()
        MPI.COMM_WORLD.Bcast(root_uniform, root=0)
        if rank != 0:
            assert local_uniform[0] != root_uniform[0]


def launch(env_name, logdir, n_epochs, num_cpu, seed, replay_strategy, policy_save_interval, clip_return,
    override_params={}, save_policies=True, gep_memory=None, buffer_location=None, study='HER', active_goal=False, **kwargs):

    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        whoami = mpi_fork(num_cpu)
        if whoami == 'parent':
            sys.exit(0)
        import baselines.common.tf_util as U
        U.single_threaded_session().__enter__()
    rank = MPI.COMM_WORLD.Get_rank()

    logger.info('RANK: ', str(rank))

    # Configure logging
    if rank == 0:
        if logdir or logger.get_dir() is None:
            logger.configure(dir=logdir)
    else:
        logger.configure()

    logdir = logger.get_dir()
    assert logdir is not None
    os.makedirs(logdir, exist_ok=True)

    # # Seed everything.
    rank_seed = seed + 1000000 * rank
    set_global_seeds(rank_seed)

    # Prepare params.
    params = config.DEFAULT_PARAMS
    params['env_name'] = env_name
    params['active_goal'] = active_goal
    params['replay_strategy'] = replay_strategy
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params.update(**override_params)  # makes it possible to override any parameter
    with open(os.path.join(logger.get_dir(), 'params.json'), 'w') as f:
        json.dump(params, f)
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    if num_cpu == 1:
        logger.warn()
        logger.warn('*** Warning ***')
        logger.warn(
            'You are running HER with just a single MPI worker. This will work, but the ' +
            'experiments that we report in Plappert et al. (2018, https://arxiv.org/abs/1802.09464) ' +
            'were obtained with --num_cpu 19. This makes a significant difference and if you ' +
            'are looking to reproduce those results, be aware of this. Please also refer to ' +
            'https://github.com/openai/baselines/issues/314 for further details.')
        logger.warn('****************')
        logger.warn()

    logger.warn('test 0 !!!!!!!!!!!!!!')

    dims = config.configure_dims(params)
    logger.warn('test 0.3 !!!!!!!!!!!!!!')
    policy = config.configure_ddpg(dims=dims, params=params, clip_return=clip_return)
    logger.warn('test 0.4 !!!!!!!!!!!!!!')
    rollout_params = {
        'exploit': False,
        'use_target_net': False,
        'use_demo_states': True,
        'compute_Q': False,
        'T': params['T'],
    }

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'use_demo_states': False,
        'compute_Q': True,
        'T': params['T'],
    }

    for name in ['T', 'rollout_batch_size', 'gamma', 'noise_eps', 'random_eps']:
        rollout_params[name] = params[name]
        eval_params[name] = params[name]

    logger.warn('test 0.5 !!!!!!!!!!!!!!')

    rollout_worker = RolloutWorker(params['make_env'], policy, dims, logger, active_goal=active_goal, **rollout_params)
    rollout_worker.seed(rank_seed)

    logger.warn('test 0.7 !!!!!!!!!!!!!!')

    evaluator = RolloutWorker(params['make_env'], policy, dims, logger, active_goal=False, **eval_params)
    evaluator.seed(rank_seed)

    logger.warn('test 1 !!!!!!!!!!!!!!')

    train(logdir=logdir, policy=policy, rollout_worker=rollout_worker, evaluator=evaluator, n_epochs=n_epochs,
          n_test_rollouts=params['n_test_rollouts'], n_cycles=params['n_cycles'], n_batches=params['n_batches'],
          policy_save_interval=policy_save_interval, save_policies=save_policies, rank=rank, gep_memory=gep_memory,
          buffer_location=buffer_location, study=study, env_name=env_name, num_cpu=num_cpu, active_goal=active_goal)

def run_her(dict_args):


    if dict_args['study'] in ['GEP_HER'] and dict_args['gep_memory'] is None:
        assert dict_args['buffer_location'] is not None
    # save parameters in data_path with pickle
    with open(dict_args['logdir']+'parameters.save', 'wb') as f:
        pickle.dump(dict_args, f)
    logger.reset()
    if MPI.COMM_WORLD.Get_rank() == 0:
        logger.configure(dir=dict_args['logdir'])
    # n_timesteps = dict_args['nb_epochs'] * dict_args['nb_epoch_cycles'] * dict_args['nb_rollout_steps']
    logger.info('Running study: '+dict_args['study']+', with noise: '+dict_args['noise_type']+', trial '+str(dict_args['trial_id']))
    # logger.info('Initialize with GEP weights? '+str(dict_args['load_initial_weights']))
    logger.info('Loading GEP memory? ' + str(dict_args['gep_memory'] is not None))

    # Run actual script.
    launch(**dict_args)

@click.command()
@click.option('--env_name', type=str, default='MalmoMountainCart-v0', help='the name of the OpenAI Gym environment that you want to train on')
@click.option('--logdir', type=str, default='./save/', help='the path to where logs and policy pickles should go. If not specified, creates a folder in /tmp/')
@click.option('--n_epochs', type=int, default=50, help='the number of training epochs to run')
@click.option('--num_cpu', type=int, default=1, help='the number of CPU cores to use (using MPI)')
@click.option('--seed', type=int, default=int(np.random.random()*1e6), help='the random seed used to seed both the environment and the training code')
@click.option('--policy_save_interval', type=int, default=5, help='the interval with which policy pickles are saved. If set to 0, only the best and latest policy will be pickled.')
@click.option('--replay_strategy', type=click.Choice(['future', 'none']), default='none', help='the HER replay strategy to be used. "future" uses HER, "none" disables HER.')
@click.option('--clip_return', type=int, default=1, help='whether or not returns should be clipped')
@click.option('--study', type=str, default='HER')  # 'DDPG'  #'GEP_PG'  #'GEP-FPG'
@click.option('--active_goal', default=False)
@click.option('--trial_id', type=int, default=0)
@click.option('--buffer_location', default=None)
@click.option('--gep_memory', default=None)
def main(**kwargs):

    dict_args = kwargs
    # # create data path
    # saving_folder = dict_args['logdir']
    # env_id = dict_args['env_name']
    # trial_id = dict_args['trial_id']
    # data_path = saving_folder + env_id + '/' + str(trial_id) + '/'
    # if os.path.exists(data_path):
    #     i = 1
    #     while os.path.exists(saving_folder + env_id + '/' + str(trial_id + 100 * i) + '/'):
    #         i += 1
    #     trial_id += i * 100
    #     print('result_path already exist, trial_id changed to: ', trial_id)
    # data_path = saving_folder + env_id + '/' + str(trial_id) + '/'
    # os.mkdir(data_path)
    # dict_args['logdir'] = data_path

    launch(**dict_args)


if __name__ == '__main__':
    main()