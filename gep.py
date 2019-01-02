from learning_module import LearningModule
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from utils.gep_utils import scale_vector, proportional_choice


class GEP(object):

    def __init__(self, config, model_babbling_mode="random", explo_noise=0.1, update_interest_step=5):
        
        self.model_babbling_mode = model_babbling_mode
        self.explo_noise = explo_noise

        # book keeping
        self.choosen_modules = [] 
        self.interests = {}

        self.policy_nb_dims = config['policy_nb_dims']
        self.modules_config = config['modules']
        print(explo_noise)
        # init learning modules
        self.modules = {}
        print('MODULES:')
        print(config['modules'])
        self.total_outcome_range = 0
        for m_name,m in self.modules_config.items():
            outcome_size = len(m['outcome_range'])
            self.modules[m_name] = LearningModule(self.policy_nb_dims, 
                                                  outcome_size, 
                                                  model_babbling_mode, 
                                                  explo_noise=explo_noise,
                                                  update_interest_step=update_interest_step)
            self.modules_config[m_name]['outcome_size'] = outcome_size
            self.total_outcome_range += outcome_size
            self.interests[m_name] = []
        print(self.total_outcome_range)
        #self.current_module = None
        self.current_policy = None

        # init main knn, will be used for exploitation
        self.knn = KNeighborsRegressor(n_neighbors=1,
                                          metric='euclidean',
                                          algorithm='ball_tree',
                                          weights='distance')
        self.knn_X = None # X = observed outcome
        self.knn_Y = None # Y = produced policies' parameters

    # returns policy parameters following and exploration process if no goal is provided
    # if a goal is provided, returns best parameter policy using NN exploitation
    def produce(self, normalized_goal=None, goal_range=None, bootstrap=False, context=None):
        if normalized_goal is not None:
            
            # use main neirest neighbor model to find best policy
            subgoal_space = self.knn_X[:,goal_range]
            #print subgoal_space.shape
            self.knn.fit(subgoal_space, self.knn_Y)
            return self.knn.predict(normalized_goal.reshape(1,-1))[0]

        if bootstrap:
            # returns random policy parameters in range [-1,1]
            self.current_policy = np.random.random(self.policy_nb_dims) * 2 - 1
            return self.current_policy


        if self.model_babbling_mode == "random":
            # random model babbling step
            module_name = np.random.choice(list(self.modules))
        elif self.model_babbling_mode == "active":
            # collect interests
            mod_name_list = list(self.modules)
            interests = np.zeros(len(mod_name_list))
            for i,(k,m) in enumerate(self.modules.items()):
                interests[i] = m.interest
            #print 'interests: %s' % interests
            # sample a module, proportionally to its interest, with 20% chance of random choice
            choosen_module_idx = proportional_choice(interests, eps=0.20)
            module_name = mod_name_list[choosen_module_idx]
            #print module_name
        else:
            return NotImplementedError

        self.choosen_modules.append(module_name) # book keeping
        module_outcome_range = self.modules_config[module_name]['outcome_range']
        module_sub_outcome = self.knn_X[:,module_outcome_range]
        #print "choosen module: %s with range: %s" % (module_name, module_outcome_range)
        #print "sub_outcome data shape:"
        #print module_sub_outcome.shape
        self.current_policy = self.modules[module_name].produce(module_sub_outcome,self.knn_Y,self.knn)

        return self.current_policy

    def perceive(self, outcome):
        #assert(outcome.shape[0] == self.total_outcome_range)
        if len(self.choosen_modules) != 0:
            # print(self.choosen_modules[-1])
            m_name = self.choosen_modules[-1]
            mod_sub_outcome = self.modules_config[m_name]['outcome_range']
            self.modules[m_name].perceive(self.current_policy,
                                          np.take(outcome, mod_sub_outcome))
        if self.model_babbling_mode == "active":
            # update interest module of choosen module if not bootstraping
            # interests book-keeping
            for m_name,m in self.modules.items():
                self.interests[m_name].append(m.interest)

        # update main knn
        # add new policy outcome pair to database
        outcome = outcome.reshape(1,-1)
        policy = self.current_policy.reshape(1,-1)
        if self.knn_X is None:
            self.knn_X = np.array(outcome)
            self.knn_Y = np.array(policy)
        else:
            self.knn_X = np.vstack((self.knn_X,outcome))
            self.knn_Y = np.vstack((self.knn_Y,policy))