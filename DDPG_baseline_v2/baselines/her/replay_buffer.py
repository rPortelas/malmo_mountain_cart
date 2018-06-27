import threading

import numpy as np


def load_from_tulip(env_id, gep_memory):
    buffer = []
    if env_id=='FetchPush-v0':
        ag_id = [3,4,5]
        dg = gep_memory['goal_representations']
        u = gep_memory['actions']
        o = gep_memory['observations']
        ag = o[:, :, ag_id]
        n_eps = ag.shape[0]
        n_t = ag.shape[1]-1
        info_is_success = np.zeros([n_eps, n_t, 1])
        g =  np.zeros([n_eps, n_t, 3])
        for i in range(n_eps):
            for j in range(n_t):
                if np.linalg.norm(dg[i, :] - ag[i, j, :], axis=-1) < 0.05:
                    info_is_success[i, j, 0] = 1
                g[i, j, :] = dg[i, :]
        # ag = ag.reshape([n_eps, 1, n_t+1, ag.shape[2]])
        # g = g.reshape([n_eps, 1, n_t, g.shape[2]])
        # info_is_success = info_is_success.reshape([n_eps, 1, n_t, info_is_success.shape[2]])
        # u = u.reshape([n_eps, 1, n_t, u.shape[2]])
        # o = o.reshape([n_eps, 1, n_t+1, o.shape[2]])
        # ag = np.repeat(ag, 2, axis=1)
        # u = np.repeat(u, 2, axis=1)
        # o = np.repeat(o, 2, axis=1)
        # g = np.repeat(g, 2, axis=1)
        # info_is_success = np.repeat(info_is_success, 2, axis=1)
        # max_buff=300
        # ind = np.random.randint(g.shape[0]-1900, g.shape[0]-1, 300)
        # buffer = dict(ag=ag[ind], u=u[ind], o=o[ind], info_is_success=info_is_success[ind], g=g[ind])
        buffer = dict(ag=ag, u=u, o=o, info_is_success=info_is_success, g=g)


    return buffer

class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions):
        """Creates a replay buffer.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        # #addition
        # self.nb_eps=0

        self.lock = threading.Lock()

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sample_transitions(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)
        # #addition
        # self.nb_eps = self.nb_eps+inc

        if inc == 1:
            idx = idx[0]
        return idx
