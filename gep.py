from learning_module import LearningModule
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from utils.gep_utils import scale_vector, proportional_choice, get_random_policy


class GEP(object):

    def __init__(self, layers, init_function_params, config, model_babbling_mode="random", explo_noise=0.1, update_interest_step=5, random_motor=0.1, interest_mean_rate=100.):
        
        self.layers = layers
        self.init_function_params = init_function_params

        self.model_babbling_mode = model_babbling_mode
        self.explo_noise = explo_noise
        self.random_motor = random_motor

        # book keeping
        self.choosen_modules = [] 
        self.interests = {}

        self.policy_nb_dims = config['policy_nb_dims']
        self.modules_config = config['modules']
        #print(explo_noise)
        # init learning modules
        self.modules = {}
        #print('MODULES:')
        #print(config['modules'])
        self.total_outcome_range = 0
        for m_name,m in self.modules_config.items():
            outcome_size = len(m['outcome_range'])
            self.modules[m_name] = LearningModule(self.policy_nb_dims,
                                                  layers,
                                                  init_function_params,
                                                  outcome_size, 
                                                  model_babbling_mode, 
                                                  explo_noise=explo_noise,
                                                  update_interest_step=update_interest_step,
                                                  mean_rate=interest_mean_rate)
            self.modules_config[m_name]['outcome_size'] = outcome_size
            self.total_outcome_range += outcome_size
            self.interests[m_name] = []
            print(outcome_size)

        #self.current_policy = None
        self.policies = []

    # returns policy parameters following and exploration process if no goal is provided
    # if a goal is provided, returns best parameter policy using NN exploitation
    def produce(self, normalized_goal=None, goal_range=None, bootstrap=False, context=None):
        add_noise = True
        if bootstrap:
            # returns random policy parameters using he_uniform
            current_policy = get_random_policy(self.layers, self.init_function_params)
            #print(self.current_policy.shape)
            return current_policy, None, False

        coin_toss = np.random.random()
        if coin_toss < self.random_motor:
            self.choosen_modules.append('random')
            current_policy = get_random_policy(self.layers, self.init_function_params)
            return current_policy, None, False

        if self.model_babbling_mode == "random":
            # random model babbling step
            module_name = np.random.choice(list(self.modules))
        elif self.model_babbling_mode == "active":
            # collect interests
            mod_name_list = list(self.modules)
            interests = np.zeros(len(mod_name_list))
            for i,(k,m) in enumerate(self.modules.items()):
                interests[i] = m.interest
            #print('interests: %s' % interests)
            # sample a module, proportionally to its interest, with 20% chance of random choice
            choosen_module_idx = proportional_choice(interests, eps=0.20)
            module_name = mod_name_list[choosen_module_idx]
            #print module_name
        else:
            return NotImplementedError

        #print("choosen module: %s with range: " % (module_name))
        self.choosen_modules.append(module_name) # book keeping
        module_outcome_range = self.modules_config[module_name]['outcome_range']
        # if module_name == 'stick1':
        #     logboy=True
        # else:
        logboy=False
        current_policy, add_noise = self.modules[module_name].produce(self.policies, logboy=logboy)
        #print(self.current_policy.shape)
        return current_policy, module_outcome_range, add_noise

    def perceive(self, outcome, policy):
        assert(outcome.shape[0] == self.total_outcome_range)
        #add data to modules
        for m_name,m in self.modules.items():
                mod_sub_outcome = self.modules_config[m_name]['outcome_range']
                #print("choosen module: %s with range: %s" % (m_name, mod_sub_outcome))
                #print("sub_outcome data shape:")
                #print(mod_sub_outcome.shape)
                m.perceive(len(self.policies), np.take(outcome, mod_sub_outcome))
                if self.model_babbling_mode == "active":
                    # interests book-keeping
                    self.interests[m_name].append(m.interest)

        if len(self.choosen_modules) != 0: # if bootstrap finished
            # update interests (or just add outcome if not active) for selected module
            # print(self.choosen_modules[-1])
            m_name = self.choosen_modules[-1]
            if m_name is not 'random':
                mod_sub_outcome = self.modules_config[m_name]['outcome_range']
                self.modules[m_name].update_interest(np.take(outcome, mod_sub_outcome))

        # store new policy
        #policy = self.current_policy
        self.policies.append(policy)

    def prepare_pickling(self):
        for m_name, m in self.modules.items():
            m.knn.nn_ready = [False, False]
            m.knn.kdtree = [None, None]
            m.knn.buffer.nn_ready = [False, False]
            m.knn.buffer.kdtree = [None, None]
            # m.knn.prepare_pickling()
            # if self.model_babbling_mode == "active":
            #     m.interest_knn.prepare_pickling()
