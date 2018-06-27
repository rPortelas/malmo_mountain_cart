from collections import deque
import itertools
import math
import numpy as np
import pickle
#from mujoco_py import MujocoException
from collections import deque

from baselines.her.util import convert_episode_to_batch_major, store_args


class RolloutWorker:

    @store_args
    def __init__(self, make_env, policy, dims, logger, T, rollout_batch_size=1,
                 exploit=False, use_target_net=False, compute_Q=False, noise_eps=0,
                 random_eps=0, history_len=100, render=False, active_goal=False, **kwargs):
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
        logger.warn('launching {} environments'.format(rollout_batch_size))
        self.envs = [make_env() for _ in range(rollout_batch_size)]
        assert self.T > 0

        self.info_keys = [key.replace('info_', '') for key in dims.keys() if key.startswith('info_')]

        self.success_history = deque(maxlen=history_len)
        self.Q_history = deque(maxlen=history_len)
        self.action_space = self.envs[0].action_space

        self._active_goal = active_goal
        ## region active_goal
        self._active_regions = False
        # self._nb_regions = 3
        # self._regions_success_counts = [BufferSuccessRates(20, self._nb_regions)] * rollout_batch_size
        # self._regions_interest = np.zeros([self._nb_regions])
        # self._regions_probabilities = softmax(self._regions_interest, 0.1)
        # self._regions_cuts = [0.44, 0.64, 0.84, 1.04]
        # self._regions_selected = [None] * rollout_batch_size

        # precision active
        self._active_precision = False
        self._epsilon = 0.05
        self._ind_eps = 0
        self._epsilons = [0.05, 0.065, 0.08, 0.095]
        self._epsilon_queues = [CompetenceQueue() for _ in self._epsilons]
        self._beta = 1
        self._epsilon_freq = [0] * len(self._epsilons)

        self.n_episodes = 0
        self.g = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # goals
        self.initial_o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        self.initial_ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        self.reset_all_rollouts()
        self.clear_history()

    def reset_rollout(self, i):
        """Resets the `i`-th rollout environment, re-samples a new goal, and updates the `initial_o`
        and `g` arrays accordingly.
        """
        if self._active_goal and self._active_regions:
            choice = np.random.choice(self._nb_regions, p=self._regions_probabilities)
            self._regions_selected[i] = choice
            bounds = np.copy(self._regions_cuts[choice: choice+2])
            ok = False
            while not ok:
                obs = self.envs[i].reset()
                # if goal within region of interest
                if obs['desired_goal'][1] <= bounds[1] and obs['desired_goal'][1] >= bounds[0]:
                    ok = True
        else:
            obs = self.envs[i].reset()
        self.initial_o[i] = obs['observation']
        if self._active_precision:
            self.initial_ag[i] = np.concatenate([obs['achieved_goal'], np.array([self._epsilon])], axis=0)
            self.g[i] = np.concatenate([obs['desired_goal'], np.array([self._epsilon])], axis=0)
        else:
            self.initial_ag[i] = obs['achieved_goal']
            self.g[i] = obs['desired_goal']

    def reset_all_rollouts(self):
        """Resets all `rollout_batch_size` rollout workers.
        """
        for i in range(self.rollout_batch_size):
            self.reset_rollout(i)

    def generate_rollouts(self):
        """Performs `rollout_batch_size` rollouts in parallel for time horizon `T` with the current
        policy acting on it accordingly.
        """
        self.reset_all_rollouts()

        # compute observations
        o = np.empty((self.rollout_batch_size, self.dims['o']), np.float32)  # observations
        ag = np.empty((self.rollout_batch_size, self.dims['g']), np.float32)  # achieved goals
        o[:] = self.initial_o
        ag[:] = self.initial_ag

        # generate episodes
        obs, achieved_goals, acts, goals, successes = [], [], [], [], []
        info_values = [np.empty((self.T, self.rollout_batch_size, self.dims['info_' + key]), np.float32) for key in self.info_keys]
        Qs = []
        for t in range(self.T):
            #self.logger.warning(str(t))
            policy_output = self.policy.get_actions(
                o, ag, self.g,
                compute_Q=self.compute_Q,
                noise_eps=self.noise_eps if not self.exploit else 0.,
                random_eps=self.random_eps if not self.exploit else 0.,
                use_target_net=self.use_target_net)

            if self.compute_Q:
                u, Q = policy_output
                Qs.append(Q)
            else:
                u = policy_output

            if u.ndim == 1:
                # The non-batched case should still have a reasonable shape.
                u = u.reshape(1, -1)
            u = (u+1) * (self.action_space.high - self.action_space.low) / 2 + self.action_space.low

            o_new = np.empty((self.rollout_batch_size, self.dims['o']))
            ag_new = np.empty((self.rollout_batch_size, self.dims['g']))
            success = np.zeros(self.rollout_batch_size)
            # compute new states and observations
            for i in range(self.rollout_batch_size):
                try:
                    # We fully ignore the reward here because it will have to be re-computed
                    # for HER.
                    curr_o_new, _, _, info = self.envs[i].step(u[i])
                    if 'is_success' in info:
                        success[i] = info['is_success']
                    o_new[i] = curr_o_new['observation']
                    if self._active_precision:
                        ag_new[i] = np.concatenate([curr_o_new['achieved_goal'], np.array([self._epsilon])], axis=0)
                    else:
                        ag_new[i] = curr_o_new['achieved_goal']

                    for idx, key in enumerate(self.info_keys):
                        info_values[idx][t, i] = info[key]
                    if self.render:
                        self.envs[i].render()
                except:
                    pass
                #     return self.generate_rollouts()

            if np.isnan(o_new).any():
                self.logger.warning('NaN caught during rollout generation. Trying again...')
                self.reset_all_rollouts()
                return self.generate_rollouts()

            obs.append(o.copy())
            achieved_goals.append(ag.copy())
            successes.append(success.copy())
            acts.append(u.copy())
            goals.append(self.g.copy())
            o[...] = o_new
            ag[...] = ag_new
        obs.append(o.copy())
        achieved_goals.append(ag.copy())
        self.initial_o[:] = o

        episode = dict(o=obs,
                       u=acts,
                       g=goals,
                       ag=achieved_goals)
        for key, value in zip(self.info_keys, info_values):
            episode['info_{}'.format(key)] = value

        # if active goal sampling, update interest values
        if self._active_goal and self._active_regions:
            for i in range(self.rollout_batch_size):
                region = self._regions_selected[i]
                s = np.array(successes)[-1, i]
                self._regions_success_counts[i].update(region=region, value=s)
                buff = self._regions_success_counts[i]._buffer[region]
                len_buffer = len(buff)
                if len_buffer >= 2:
                    half_len_buffer = int(len_buffer / 2)
                    self._regions_interest[region] = np.mean(buff[-half_len_buffer:]) - np.mean(buff[:half_len_buffer])
            self._regions_probabilities = softmax(u=self._regions_interest, t=0.1)  # 0 and 1 success rate gives 0.1 and 0.9 prob, use 0.72 for 0.2/0.8
            self._regions_probabilities[0] = 1 - self._regions_probabilities[1:].sum()

        if self._active_goal and self._active_precision:
            succ_eps = np.zeros([self.rollout_batch_size])
            for i in range(self.rollout_batch_size):
                ach_g = achieved_goals[-1][i]
                des_g = goals[-1][i]
                succ_eps[i] = self.envs[i].compute_reward(ach_g.reshape(1,-1), des_g.reshape(1,-1), None)
                self._epsilon_queues[self._ind_eps].append(point = int(succ_eps[i]==0))
            self._ind_eps, self._epsilon = self.sample_epsilon()
            self._epsilon_freq[self._ind_eps] += 1

        # stats
        successful = np.array(successes)[-1, :]
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

    def sample_epsilon(self):
        sizes = [queue.size for queue in self._epsilon_queues]
        if np.all(np.array(sizes)>50):
            CPs = [math.pow(queue.CP, self._beta) for queue in self._epsilon_queues]
            probas = CPs / np.sum(CPs)
            idx = np.random.choice(range(len(self._epsilons)), p=probas)
        else:
            idx = np.random.randint(0,len(self._epsilon_queues))
        return idx, self._epsilons[idx]

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
        if self._active_precision:
            logs += [('list_freq_eps'+str(self._epsilons[i]), self.list_freq[i]) for i in range(len(self._epsilons))]
            logs += [('list_comp_eps'+str(self._epsilons[i]), self.list_comp[i]) for i in range(len(self._epsilons))]
            logs += [('list_CP_eps'+str(self._epsilons[i]), self.list_CP[i]) for i in range(len(self._epsilons))]

        if prefix is not '' and not prefix.endswith('/'):
            return [(prefix + '/' + key, val) for key, val in logs]
        else:
            return logs

    def seed(self, seed):
        """Seeds each environment with a distinct seed derived from the passed in global seed.
        """
        for idx, env in enumerate(self.envs):
            env.seed(seed + 1000 * idx)

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def list_CP(self):
        return [float("{0:.3f}".format(self._epsilon_queues[idx].CP)) for idx in range(len(self._epsilons))]

    @property
    def list_comp(self):
        return [float("{0:.3f}".format(self._epsilon_queues[idx].competence)) for idx in range(len(self._epsilons))]

    @property
    def list_freq(self):
        return self._epsilon_freq





class CompetenceQueue():
    def __init__(self, window = 50):
        self.window = window
        self.points = deque(maxlen=2 * self.window)
        self.CP = 0.001
        self.competence = 0.001

    def update_CP(self):
        if self.size > self.window: # this allows the keep uniform proba on epsiolon until some information is gathered.
            window = min(self.size // 2, self.window)
            q1 = list(itertools.islice(self.points, self.size - window, self.size))
            q2 = list(itertools.islice(self.points, self.size - 2 * window, self.size - window))
            self.CP = max(np.abs(np.sum(q1) - np.sum(q2)) / (2 * window), 0.001)
            self.competence = np.sum(q1) / window

    def append(self, point):
        self.points.append(point)
        self.update_CP()

    @property
    def size(self):
        return len(self.points)

    @property
    def full(self):
        return self.size == self.points.maxlen

