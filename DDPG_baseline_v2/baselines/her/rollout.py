from collections import deque

import numpy as np
import pickle
#from mujoco_py import MujocoException

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, **kwargs):
        """Rollout worker generates experience by interacting with one or many environments.

        Args:
            make_env (function): a factory function that creates a new instance of the environment
                when called
            policy (object): the policy that is used to act
            dims (dict of ints): the dimensions for observations (o), goals (g), and actions (u)
            logger (object): the logger that is used by the rollout worker
            rollout_batch_size (int): the number of parallel rollouts that should be used
            exploit (boolean): whether or not to exploit, i.e. to act optimally according to the
                current policy without any exploration
            use_target_net (boolean): whether or not to use the target net for rollouts
            compute_Q (boolean): whether or not to compute the Q values alongside the actions
            noise_eps (float): scale of the additive Gaussian noise
            random_eps (float): probability of selecting a completely random action
            history_len (int): length of history for statistics smoothing
            render (boolean): whether or not to render the rollouts
        """
        if exploit == True:
            self.env = kwargs['rollout_envs']
        else:
            self.env = make_env()
        assert self.T > 0


        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)

        self.n_episodes = 0
        self.g = np.empty((1, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((1, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((1, self.dims['g']), np.float32)  # achieved goals
        self.reset_rollout()
        self.clear_history()
        self.seed_idx = 0

    def reset_rollout(self):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        obs = self.env.reset()
        self.initial_o = obs['observation']
        self.initial_ag = obs['achieved_goal']
        self.g = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout()

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
                policy acting on it accordingly.
                """

        #print('dims: {}'.format(list(self.dims)))
        # generate episodes
        #obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        obs = np.empty((self.T + 1, self.rollout_batch_size, self.dims['o']), np.float32)
        achieved_goals = np.empty((self.T + 1, self.rollout_batch_size, self.dims['g']), np.float32)
        acts = np.empty((self.T, self.rollout_batch_size, self.dims['u']), np.float32)
        goals = np.empty((self.T, self.rollout_batch_size, self.dims['g']), np.float32)
        successes = np.empty((self.T, self.rollout_batch_size, 1), np.bool)
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in
                       self.info_keys]
        Qs = []

        for i in range(self.rollout_batch_size):
            self.reset_rollout()
            # compute observations
            o = np.empty((1, self.dims['o']), np.float32)  # observations
            ag = np.empty((1, self.dims['g']), np.float32)  # achieved goals
            o[:] = self.initial_o
            ag[:] = self.initial_ag
            for t in range(self.T):
                policy_output = self.policy.get_actions(
                    o[:], ag[:], self.g[:],
                    compute_Q=self.compute_Q,
                    noise_eps=self.noise_eps if not self.exploit else 0.,
                    random_eps=self.random_eps if not self.exploit else 0.,
                    use_target_net=self.use_target_net)
                #print('ACTION SHAPE:')
                #print(policy_output.shape)
                #print(policy_output)

                if self.compute_Q:
                    u, Q = policy_output
                    Qs.append(Q)
                else:
                    u = policy_output

                if u.ndim == 1:
                    # The non-batched case should still have a reasonable shape.
                    u = u.reshape(1, -1)

                o_new = np.empty((1, self.dims['o']))
                #print('O_new before {}'.format(o_new.shape))
                ag_new = np.empty((1, self.dims['g']))
                # compute new states and observations
                #              try:
                # We fully ignore the reward here because it will have to be re-computed
                # for HER.
                curr_o_new, _, _, info = self.env.step(u[0])
                if 'is_success' in info:
                    success = info['is_success']
                o_new[:] = curr_o_new['observation']
                ag_new[:] = curr_o_new['achieved_goal']
                #print('O_new after {}'.format(o_new.shape))
                for idx, key in enumerate(self.info_keys):
                    info_values[idx][t, i] = info[key]
                if self.render:
                    self.env.render()

                if np.isnan(o_new).any():
                    self.logger.warning('NaN caught during rollout generation. Trying again...')
                    self.reset_rollout()
                    return self.generate_rollouts()

                obs[t, i, :] = o.copy()
                achieved_goals[t, i, :] = ag.copy()
                successes[t, i, :] = success
                acts[t, i, :] = u.copy()
                goals[t, i, :] = self.g.copy()

                #print('O before {}'.format(o.shape))
                o[...] = o_new
                #print('O after {}'.format(o.shape))
                ag[...] = ag_new
                #print(t)
            #print('final t: {}'.format(t))
            obs[t+1, i, :] = o.copy()
            achieved_goals[t+1, i, :] = ag.copy()
            #self.initial_o[:] = o

        episode = dict(o=list(obs),
                       u=list(acts),
                       g=list(goals),
                       ag=list(achieved_goals))
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # stats
        successful = successes[-1, :, 0]
        #print('succful shp: {}'.format(successful.shape))
        assert successful.shape == (self.rollout_batch_size,)
        success_rate = np.mean(successful)
        self.success_history.append(success_rate)
        if self.compute_Q:
            self.Q_history.append(np.mean(Qs))
        self.n_episodes += self.rollout_batch_size

        return convert_episode_to_batch_major(episode)


    def clear_history(self):
        """Clears all histories that are used for statistics
        """
        self.success_history.clear()
        self.Q_history.clear()

    def current_success_rate(self):
        return np.mean(self.success_history)

    def current_mean_Q(self):
        return np.mean(self.Q_history)

    def save_policy(self, path):
        """Pickles the current policy for later inspection.
        """
        with open(path, 'wb') as f:
            pickle.dump(self.policy, f)

    def logs(self, prefix='worker'):
        """Generates a dictionary that contains all collected statistics.
        """
        logs = []
        logs += [('success_rate', np.mean(self.success_history))]
        if self.compute_Q:
            logs += [('mean_Q', np.mean(self.Q_history))]
        logs += [('episode', self.n_episodes)]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        self.env.seed(seed + self.seed_idx * 1000 )
        self.seed_idx += 1
