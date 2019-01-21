import numpy as np
#from sklearn.neighbors import KNeighborsRegressor
#from sklearn.neighbors import NearestNeighbors
from utils.gep_utils import scale_vector
import scipy.spatial


class LearningModule(object):
    # outcome_bounds must be a 2d array with column 1 = mins and column 2 = maxs
    def __init__(self, policy_nb_dims, outcome_size, babbling_mode, explo_noise=0.1, n_neighbors=1,
                 interest_running_mean=200, update_interest_step=5):
        self.policy_nb_dims = policy_nb_dims
        self.o_size = outcome_size
        self.explo_noise = explo_noise
        self.babbling_mode = babbling_mode

        self.generated_goals = None
        self.observed_outcomes = None

        if self.babbling_mode == "active":
            self.mean_rate = 100. # running mean window
            self.interest = 0
            self.progress = 0
            #self.interest_knn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='ball_tree')
            #self.interest_knn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='ball_tree')
            self.update_interest_step = update_interest_step # 4 exploration for 1 exploitation
            self.counter = 0

    # sample a goal in outcome space and find closest neighbor in (param,outcome) database
    # RETURN policy param with added gaussian noise
    def produce(self, outcomes, policies):
        # draw randow goal in bounded outcome space
        goal = np.random.random(self.o_size) * 2 - 1
        goal = goal.reshape(1,-1)
        add_noise = True

        
        if self.babbling_mode == "active":
            #print self.counter
            self.counter += 1
            if self.update_interest_step == 1: #compute noisy interest at every step
                add_noise = True
            elif (self.counter % self.update_interest_step) == 0: #exploitation step
                add_noise = False
                if self.generated_goals is None:
                    self.generated_goals = np.array(goal)
                else:
                    self.generated_goals = np.vstack((self.generated_goals, goal))



        # get closest outcome in database and retreive corresponding policy
        #knn.fit(outcomes, policies)
        #policy = knn.predict(goal)
        knn = scipy.spatial.cKDTree(outcomes, balanced_tree=False, compact_nodes=False)
        dist, ind = knn.query(goal, k=1, distance_upper_bound=np.inf, eps=0.0, p=2)
        policy = policies[ind]
        # add gaussian noise for exploration
        if add_noise:
            #print 'adding noise'
            policy += np.random.normal(0, self.explo_noise, self.policy_nb_dims)
            policy = np.clip(policy, -1, 1)
        return policy[0]

    def perceive(self, policy, outcome):
        policy = policy.reshape(1,-1)
        outcome = outcome.reshape(1,-1)
        if self.babbling_mode == "active":
            # update interest, only if:
            # - not in bootstrap phase since no goal is generated during this phase
            # - not in an exploration phase (update progress when exploiting for better accuracy)
            if (self.generated_goals is not None) and self.generated_goals.shape[0] < 3 and (self.counter % self.update_interest_step) == 0:
                if self.generated_goals.shape[0] == 1:
                    self.observed_outcomes = np.array(outcome)
                else:
                    self.observed_outcomes = np.vstack((self.observed_outcomes, outcome))
                return
            elif (self.counter % self.update_interest_step) == 0:
                self.counter = 0 # reset counter
                #print 'updating interest'
                #print 'update'gene
                assert(self.generated_goals.shape[0] == (self.observed_outcomes.shape[0] + 1))
                previous_goals = self.generated_goals[:-1,:]
                current_goal = self.generated_goals[-1,:].reshape(1,-1)
                #print 'current_generated_goal: %s, with shape: %s' % (current_goal,current_goal.shape)
                #print 'previous_generated_goal: %s, with shape: %s' % (previous_goals,previous_goals.shape)
                # find closest previous goal to current goal

                #self.interest_knn.fit(previous_goals, self.observed_outcomes)
                #dist, idx = self.interest_knn.kneighbors(current_goal)
                #closest_previous_goal = previous_goals[idx[0]]
                #closest_previous_goal_outcome = self.observed_outcomes[idx[0],:]
                interest_knn = scipy.spatial.cKDTree(previous_goals, balanced_tree=False, compact_nodes=False)
                dist, idx = interest_knn.query(current_goal, k=1, distance_upper_bound=np.inf, eps=0.0, p=2)
                closest_previous_goal = previous_goals[idx]
                
                #print 'closest previous goal is index:%s, val: %s' % (idx[0], closest_previous_goal)
                # retrieve old outcome corresponding to closest previous goal
                closest_previous_goal_outcome = self.observed_outcomes[idx,:]

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
                self.observed_outcomes = np.vstack((self.observed_outcomes, outcome))
            else:
                pass
        else:
            pass