import numpy as np
from sklearn.neighbors import KNeighborsRegressor

class BufferedBalltree(object):
    def __init__(self, buffer_size=1000):
        # init main knn, will be used for exploitation
        self.knn = KNeighborsRegressor(n_neighbors=1,
                                       metric='euclidean',
                                       algorithm='ball_tree',
                                       weights='distance')
        self.knn_X = None  # X = observed outcome
        self.knn_Y = None  # Y = produced policies' parameters

        self.buffer_knn = KNeighborsRegressor(n_neighbors=1,
                                       metric='euclidean',
                                       algorithm='ball_tree',
                                       weights='distance')
        self.buffer_knn_X = None  # X = observed outcome
        self.buffer_knn_Y = None  # Y = produced policies' parameters

        self.buffer_size = buffer_size

    def add(self, outcome, policy):
        # add to buffer
        if self.buffer_knn_X is None:
            self.buffer_knn_X = np.array(outcome)
            self.buffer_knn_Y = np.array(policy)
        else:
            self.buffer_knn_X = np.vstack((self.buffer_knn_X, outcome))
            self.buffer_knn_Y = np.vstack((self.buffer_knn_Y, policy))

        # re-fit big tree if buffer is full
        if len(self.buffer_knn_X) >= self.buffer_size:
            #empty buffer to main tree
            if self.knn_X is None:
                self.knn_X = self.buffer_knn_X
                self.knn_Y = self.buffer_knn_Y
            else:
                self.knn_X = np.vstack((self.knn_X, self.buffer_knn_X))
                self.knn_Y = np.vstack((self.knn_Y, self.buffer_knn_Y))

            # reset buffer
            self.buffer_knn_X = None
            self.buffer_knn_Y = None

            # fit main knn
            self.knn.fit(self.knn_X, self.knn_Y)
        else: #just fit buffer
            self.buffer_knn.fit(self.buffer_knn_X, self.buffer_knn_Y)

    def predict(self, goal):
        if self.knn_X is None:
            return self.buffer_knn.predict(goal)[0]
        elif self.buffer_knn_X is None:
            return self.knn.predict(goal)[0]
        else:
            # print('goal is {}'.format(goal))
            best_d_buffer, ind_buf = self.buffer_knn.kneighbors(goal)
            best_d_main, ind_main = self.knn.kneighbors(goal)
            ind_buf = ind_buf[0][0]
            ind_main = ind_main[0][0]
            # print(len(self.knn_X))
            # print(len(self.buffer_knn_X))
            # print(best_d_buffer)
            # print(ind_buf)
            # print(self.buffer_knn_X[ind_buf])
            # print(best_d_main)
            # print(ind_main)
            # print(self.knn_X[ind_main])
            if best_d_buffer <= best_d_main:
                #print("best is buff")
                #print(self.buffer_knn_Y[ind_buf].shape)
                #print('hh')
                return self.buffer_knn_Y[ind_buf]
            else:
                #print("best_isknn")
                #print(self.knn_Y[ind_main].shape)
                #print('gg')
                return self.knn_Y[ind_main]

    #def get_outcomes():
    #    if self.knn_X is None:
    #        return self.buffer_knn_X
    #    elif self.buffer_knn_X is None:
    #        return self.knn_X
    #    else:
    #        return np.vstack((self.knn_X, self.buffer_knn_X))


