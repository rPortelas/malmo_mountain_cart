import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.spatial
import time
import pickle


class BufferedcKDTree(object):
    def __init__(self, buffer_size=1000):
        # init main knn, will be used for exploitation
        self.knn = None
        self.knn_X = [] # X = observed outcome

        self.buffer_knn = None
        self.buffer_knn_X = []  # X = observed outcome

        self.buffer_size = buffer_size
        self.main_is_ready = False
        self.buffer_is_ready = False

    def add(self, outcome):
        # add to buffer
        self.buffer_knn_X.append(outcome)
        self.buffer_is_ready = False

        # re-fit big tree if buffer is full
        if len(self.buffer_knn_X) >= self.buffer_size:
            # empty buffer to main tree
            self.knn_X = self.knn_X + self.buffer_knn_X
            self.main_is_ready = False
            # reset buffer
            self.buffer_knn_X = []

    def fit_buffer(self):
        self.buffer_knn = scipy.spatial.cKDTree(np.array(self.buffer_knn_X), balanced_tree=False, compact_nodes=False)

    def fit_main(self):
        self.knn = scipy.spatial.cKDTree(np.array(self.knn_X), balanced_tree=False, compact_nodes=False)

    def predict(self, goal):

        if len(self.knn_X) == 0:
            if not self.buffer_is_ready:
                self.fit_buffer()
                self.buffer_is_ready = True
            return self.buffer_knn.query(goal)
        elif len(self.buffer_knn_X) == 0:
            if not self.main_is_ready:
                # fit main knn
                self.fit_main()
                self.main_is_ready = True
            return self.knn.query(goal)
        else:
            if not self.main_is_ready:
                self.fit_main()
                self.main_is_ready = True
            if not self.buffer_is_ready:
                self.fit_buffer()
                self.buffer_is_ready = True
            best_d_buffer, ind_buf = self.buffer_knn.query(goal)
            best_d_main, ind_main = self.knn.query(goal)

            if best_d_buffer <= best_d_main:
                return best_d_buffer, ind_buf
            else:
                return best_d_main, ind_main

    def prepare_pickling(self):
        # cKDtree cannot be pickled
        self.buffer_is_ready = False
        self.main_is_ready = False
        self.knn = None
        self.buffer_knn = None

class BufferedBalltree(object):
    def __init__(self, buffer_size=1000):
        # init main knn, will be used for exploitation
        self.knn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='ball_tree')
        self.knn_X = []  # X = observed outcome

        self.buffer_knn = NearestNeighbors(n_neighbors=1, metric='euclidean', algorithm='ball_tree')
        self.buffer_knn_X = [] # X = observed outcome

        self.buffer_size = buffer_size

    def add(self, outcome):
        # add to buffer
        self.buffer_knn_X.append(outcome)

        # re-fit big tree if buffer is full
        if len(self.buffer_knn_X) >= self.buffer_size:
            #empty buffer to main tree
            self.knn_X = self.knn_X + self.buffer_knn_X

            # reset buffer
            self.buffer_knn_X = []

            # fit main knn
            self.knn.fit(self.knn_X)
        else: #just fit buffer
            self.buffer_knn.fit(self.buffer_knn_X)

    def predict(self, goal):
        if len(self.knn_X) == 0:
            return self.buffer_knn.kneighbors(goal)[1][0][0]
        elif len(self.buffer_knn_X) == 0:
            return self.knn.kneighbors(goal)[1][0][0]
        else:
            # print('goal is {}'.format(goal))
            best_d_buffer, ind_buf = self.buffer_knn.kneighbors(goal)
            best_d_main, ind_main = self.knn.kneighbors(goal)
            ind_buf = ind_buf[0][0]
            ind_main = ind_main[0][0]
            assert(len(best_d_buffer[0]) == 1)
            assert (len(best_d_main[0]) == 1)
            # print(len(self.knn_X))
            # print(len(self.buffer_knn_X))
            # print(best_d_buffer)
            # print(ind_buf)
            # print(self.buffer_knn_X[ind_buf])
            # print(best_d_main)
            # print(ind_main)
            # print(self.knn_X[ind_main])
            if best_d_buffer[0][0] <= best_d_main[0][0]:
                #print("best is buff")
                #print(self.buffer_knn_Y[ind_buf].shape)
                #print('hh')
                return ind_buf
            else:
                #print("best_isknn")
                #print(self.knn_Y[ind_main].shape)
                #print('gg')
                return ind_main

    #def get_outcomes():
    #    if self.knn_X is None:
    #        return self.buffer_knn_X
    #    elif self.buffer_knn_X is None:
    #        return self.knn_X
    #    else:
    #        return np.vstack((self.knn_X, self.buffer_knn_X))


