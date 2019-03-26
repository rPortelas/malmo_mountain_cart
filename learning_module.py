import numpy as np
from utils.gep_utils import get_random_policy
from utils.dataset import BufferedDataset
import copy


class LearningModule(object):
    # outcome_bounds must be a 2d array with column 1 = mins and column 2 = maxs
    def __init__(self, policy_nb_dims, layers, init_function_params, outcome_size,
                 babbling_mode, explo_noise=0.1, update_interest_step=5, mean_rate=100.):
        self.policy_nb_dims = policy_nb_dims
        self.layers = layers
        self.init_function_params = init_function_params
        self.o_size = outcome_size
        self.explo_noise = explo_noise
        self.babbling_mode = babbling_mode

        self.generated_goals = []
        self.observed_outcomes = []

        if self.babbling_mode == "active":
            self.mean_rate = mean_rate  # running mean window
            self.interest = 0
            self.progress = 0
            self.interest_knn = BufferedDataset(self.o_size, self.o_size, buffer_size=200, lateness=0)
            self.update_interest_step = update_interest_step  # default is 4 exploration for 1 exploitation
            self.counter = 0

        self.knn = BufferedDataset(1, self.o_size, buffer_size=1000, lateness=0)  # use index instead of policies

    # sample a goal in outcome space and find closest neighbor in (param,outcome) database
    # RETURN policy param with added gaussian noise
    def produce(self, policies, goal=None):
        if goal:  # test time, no noise
            _, policy_idx = self.knn.nn_y(goal)
            policy = copy.deepcopy(policies[policy_idx[0]])
            return policy, False

        # draw randow goal in bounded outcome space
        goal = np.random.random(self.o_size) * 2 - 1
        goal = goal
        add_noise = True

        if self.babbling_mode == "active":
            self.counter += 1
            if self.update_interest_step == 1:  # compute noisy interest at every step
                add_noise = True
            elif (self.counter % self.update_interest_step) == 0:  # exploitation step
                add_noise = False
                self.generated_goals.append(goal)

        # get closest outcome in database and retreive corresponding policy
        _, policy_idx = self.knn.nn_y(goal)

        policy_knn_idx = self.knn.get_x(policy_idx[0])
        assert(policy_idx[0] == policy_knn_idx)
        policy = copy.deepcopy(policies[policy_idx[0]])

        # add gaussian noise for exploration
        if add_noise:
            if policy_idx[0] == 0:  # the first ever seen is the best == we found nothing, revert to random motor
                policy = get_random_policy(self.layers, self.init_function_params)
                add_noise = False
            else:
                pass  # noise will be added at run time
        return policy, add_noise

    def perceive(self, policy_idx, outcome):  # must be called for each episode
        # add to knn
        self.knn.add_xy(policy_idx, outcome)

    def update_interest(self, outcome):  # must be called only if module is selected
        if self.babbling_mode == "active":
            # update interest, only if:
            # - not in bootstrap phase since no goal is generated during this phase
            # - not in an exploration phase (update progress when exploiting for better accuracy)
            if len(self.generated_goals) < 3 and ((self.counter % self.update_interest_step) == 0):
                self.interest_knn.add_xy(outcome, self.generated_goals[-1])
                if (self.counter % self.update_interest_step) == 0:
                    self.counter = 0  # reset counter
                return
            elif (self.counter % self.update_interest_step) == 0:
                self.counter = 0  # reset counter
                current_goal = self.generated_goals[-1]
                # find closest previous goal to current goal
                dist, idx = self.interest_knn.nn_y(current_goal)
                # retrieve old outcome corresponding to closest previous goal
                closest_previous_goal_outcome = self.interest_knn.get_x(idx[0])

                # compute Progress as dist(s_g,s') - dist(s_g,s)
                # with s_g current goal and s observed outcome
                # s_g' closest previous goal and s' its observed outcome
                dist_goal_old_outcome = np.linalg.norm(current_goal - closest_previous_goal_outcome) / self.o_size
                dist_goal_cur_outcome = np.linalg.norm(current_goal - outcome) / self.o_size
                progress = dist_goal_old_outcome - dist_goal_cur_outcome
                self.progress = ((self.mean_rate-1)/self.mean_rate) * self.progress + (1/self.mean_rate) * progress
                self.interest = np.abs(self.progress)
                self.interest_knn.add_xy(outcome, self.generated_goals[-1])
            else:
                pass
