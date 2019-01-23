import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from utils.gep_utils import scale_vector
from utils.gep_utils import get_random_policy
#from utils.knn_variants import BufferedcKDTree
from utils.dataset import BufferedDataset
import copy


class LearningModule(object):
    # outcome_bounds must be a 2d array with column 1 = mins and column 2 = maxs
    def __init__(self, policy_nb_dims, layers, init_function_params, outcome_size, babbling_mode, explo_noise=0.1, update_interest_step=5, mean_rate=100.):
        self.policy_nb_dims = policy_nb_dims
        self.layers = layers
        self.init_function_params = init_function_params
        self.o_size = outcome_size
        self.explo_noise = explo_noise
        self.babbling_mode = babbling_mode

        self.generated_goals = []
        self.observed_outcomes = []
        self.LOG = False

        if self.babbling_mode == "active":
            self.mean_rate = mean_rate # running mean window
            self.interest = 0
            self.progress = 0
            self.interest_knn = BufferedDataset(self.o_size, self.o_size, buffer_size=200, lateness=0)
            self.update_interest_step = update_interest_step # 4 exploration for 1 exploitation
            self.counter = 0


        self.knn = BufferedDataset(1, self.o_size, buffer_size=1000, lateness=0) #use index instead of policies
        #self.tmp_outcomes = []

    # sample a goal in outcome space and find closest neighbor in (param,outcome) database
    # RETURN policy param with added gaussian noise
    def produce(self, policies, logboy=False):
        # draw randow goal in bounded outcome space
        goal = np.random.random(self.o_size) * 2 - 1
        goal = goal
        if self.LOG: print("goal is {} {}".format(goal[0:3], goal.shape))
        add_noise = True


        if self.babbling_mode == "active":
            #print self.counter
            self.counter += 1
            if self.update_interest_step == 1: #compute noisy interest at every step
                add_noise = True
            elif (self.counter % self.update_interest_step) == 0: #exploitation step
                add_noise = False
                self.generated_goals.append(goal)


        # get closest outcome in database and retreive corresponding policy
        _, policy_idx = self.knn.nn_y(goal)

        #if logboy: print("nb:{} val:{}".format(policy_idx, policies[policy_idx[0]][155]))

        #policy = policies[policy_idx]
        policy_knn_idx = self.knn.get_x(policy_idx[0])
        if logboy: print(policy_knn_idx)
        assert(policy_idx[0] == policy_knn_idx)
        policy = copy.deepcopy(policies[policy_idx[0]])


        # add gaussian noise for exploration
        if add_noise:

            if policy_idx[0] == 0:  # the first ever seen is the best == we found nothing, revert to random motor
                if logboy: print("{} reveeeeert".format(policy_idx))
                policy = get_random_policy(self.layers, self.init_function_params)
            else:
                #if logboy: print("{} old".format(policy_idx))
                if self.LOG: print('adding noise: {} on {}'.format(self.explo_noise, policy[0][200]))
                gaussian_noise = np.random.normal(0, self.explo_noise, self.policy_nb_dims)
                for i in range(len(policy)):
                    if self.LOG: print("before {}:{}".format(i,policy[i][200]))
                    policy[i] += gaussian_noise
                    policy[i] = np.clip(policy[i], -1, 1)
                #print("{}=={}".format(policy.shape, policy[200:210]))
                    if self.LOG: print("after {}=={}".format(policy[i].shape, policy[i][200]))
                #print('done')
        if logboy: print("noise: {} {}: before: {}, after: {}, ({})".format(add_noise, self.counter, policies[policy_idx[0]][0][155], policy[0][155], self.explo_noise))
        return policy

    def perceive(self, policy_idx, outcome): # must be called for each episode
        # add to knn
        #self.knn.add(outcome)
        self.knn.add_xy(policy_idx, outcome)
        #self.tmp_outcomes.append(outcome)
        #check if correctly organized
        # for i in range(len(self.tmp_outcomes)):
        #     knn_out = self.knn.get_x(i)
        #     tmp_out = self.tmp_outcomes[i]
        #     assert((knn_out == tmp_out).all())
        #     print(i)

    def update_interest(self, outcome): # must be called only if module is selected
        if self.babbling_mode == "active":
            # update interest, only if:
            # - not in bootstrap phase since no goal is generated during this phase
            # - not in an exploration phase (update progress when exploiting for better accuracy)
            if len(self.generated_goals) < 3 and ((self.counter % self.update_interest_step) == 0):
                #self.interest_knn.add(self.generated_goals[-1])
                self.interest_knn.add_xy(outcome, self.generated_goals[-1])
                if ((self.counter % self.update_interest_step) == 0):
                    self.counter = 0  # reset counter
                #self.observed_outcomes.append(outcome)
                return
            elif ((self.counter % self.update_interest_step) == 0):
                self.counter = 0 # reset counter
                #print 'updating interest'
                #print 'update'gene
                #assert(len(self.generated_goals) == (len(self.observed_outcomes) + 1))
                #previous_goals = self.generated_goals[:-1]
                current_goal = self.generated_goals[-1]
                #print 'current_generated_goal: %s, with shape: %s' % (current_goal,current_goal.shape)
                #print 'previous_generated_goal: %s, with shape: %s' % (previous_goals,previous_goals.shape)
                # find closest previous goal to current goal
                dist, idx = self.interest_knn.nn_y(current_goal)
                #closest_previous_goal = previous_goals[idx]
                closest_previous_goal = self.interest_knn.get_y(idx[0])
                closest_previous_goal_outcome = self.interest_knn.get_x(idx[0])
                #print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)
                # retrieve old outcome corresponding to closest previous goal
                #closest_previous_goal_outcome = self.observed_outcomes[idx]

                # compute Progress as dist(s_g,s') - dist(s_g,s)
                # with s_g current goal and s observed outcome
                # s_g' closest previous goal and s' its observed outcome
                #print 'old interest: %s' % self.interest
                dist_goal_old_outcome = np.linalg.norm(current_goal - closest_previous_goal_outcome) / self.o_size
                dist_goal_cur_outcome = np.linalg.norm(current_goal - outcome) / self.o_size
                progress = dist_goal_old_outcome - dist_goal_cur_outcome
                self.progress = ((self.mean_rate-1)/self.mean_rate) * self.progress + (1/self.mean_rate) * progress
                self.interest = np.abs(self.progress)

                #update observed outcomes
                #self.observed_outcomes.append(outcome)
                #self.interest_knn.add(self.generated_goals[-1])
                self.interest_knn.add_xy(outcome, self.generated_goals[-1])
            else:
                pass
