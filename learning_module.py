import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from utils.gep_utils import scale_vector
from utils.knn_variants import BufferedcKDTree


class LearningModule(object):
    # outcome_bounds must be a 2d array with column 1 = mins and column 2 = maxs
    def __init__(self, policy_nb_dims, outcome_size, babbling_mode, explo_noise=0.1, n_neighbors=1,
                 interest_running_mean=200, update_interest_step=5):
        self.policy_nb_dims = policy_nb_dims
        self.o_size = outcome_size
        self.explo_noise = explo_noise
        self.babbling_mode = babbling_mode

        self.generated_goals = []
        self.observed_outcomes = []

        if self.babbling_mode == "active":
            self.mean_rate = 100. # running mean window
            self.interest = 0
            self.progress = 0
            self.interest_knn = BufferedcKDTree()
            self.update_interest_step = update_interest_step # 4 exploration for 1 exploitation
            self.counter = 0


        self.knn = BufferedcKDTree()

    # sample a goal in outcome space and find closest neighbor in (param,outcome) database
    # RETURN policy param with added gaussian noise
    def produce(self, policies):
        # draw randow goal in bounded outcome space
        goal = np.random.random(self.o_size) * 2 - 1
        goal = goal
        add_noise = True

        self.generated_goals.append(goal)

        if self.babbling_mode == "active":
            #print self.counter
            self.counter += 1
            if self.update_interest_step == 1: #compute noisy interest at every step
                add_noise = True
            elif (self.counter % self.update_interest_step) == 0: #exploitation step
                add_noise = False


        # get closest outcome in database and retreive corresponding policy
        _, policy_idx = self.knn.predict(goal)
        policy = policies[policy_idx]

        # add gaussian noise for exploration
        if add_noise:
            #print 'adding noise'
            policy += np.random.normal(0, self.explo_noise, self.policy_nb_dims)
            policy = np.clip(policy, -1, 1)

        return policy

    def perceive(self, outcome): # must be called for each episode
        outcome = outcome
        # add to knn
        self.knn.add(outcome)

    def update_interest(self, outcome): # must be called only if module is selected
        if self.babbling_mode == "active":
            # update interest, only if:
            # - not in bootstrap phase since no goal is generated during this phase
            # - not in an exploration phase (update progress when exploiting for better accuracy)
            if len(self.generated_goals) < 3 or (self.counter % self.update_interest_step) != 0:
                self.interest_knn.add(self.generated_goals[-1])
                self.observed_outcomes.append(outcome)
                return
            self.counter = 0 # reset counter
            #print 'updating interest'
            #print 'update'gene
            assert(len(self.generated_goals) == (len(self.observed_outcomes) + 1))
            previous_goals = self.generated_goals[:-1]
            current_goal = self.generated_goals[-1]
            #print 'current_generated_goal: %s, with shape: %s' % (current_goal,current_goal.shape)
            #print 'previous_generated_goal: %s, with shape: %s' % (previous_goals,previous_goals.shape)
            # find closest previous goal to current goal
            dist, idx = self.interest_knn.predict(current_goal)
            closest_previous_goal = previous_goals[idx]
            #print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)
            # retrieve old outcome corresponding to closest previous goal
            closest_previous_goal_outcome = self.observed_outcomes[idx]

            # compute Progress as dist(s_g,s') - dist(s_g,s)
            # with s_g current goal and s observed outcome
            # s_g' closest previous goal and s' its observed outcome
            #print 'old interest: %s' % self.interest
            dist_goal_old_outcome = np.linalg.norm(current_goal - closest_previous_goal_outcome)
            dist_goal_cur_outcome = np.linalg.norm(current_goal - outcome)
            progress = dist_goal_old_outcome - dist_goal_cur_outcome
            self.progress = ((self.mean_rate-1)/self.mean_rate) * self.progress + (1/self.mean_rate) * progress
            self.interest = np.abs(self.progress)
            
            #update observed outcomes
            self.observed_outcomes.append(outcome)
            self.interest_knn.add(self.generated_goals[-1])
        else:
            if len(self.generated_goals) != 0: #end of bt phase, goals are sampled
                self.observed_outcomes.append(outcome)
