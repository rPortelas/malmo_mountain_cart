import numpy as np
import pickle
import sys
from gep_utils import *

def inside_polygon(x, y, points):
    """
    Return True if a coordinate (x, y) is inside a polygon defined by
    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].

    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
    """
    n = len(points)
    inside = False
    p1x, p1y = points[0]
    for i in range(1, n + 1):
        p2x, p2y = points[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def generate_goals_in_arena(nb_points):
    b = Bounds()
    b.add('agent_x',[288.3,294.7])
    b.add('agent_z',[433.3,443.7])
    # arena boundaries
    points = np.array([[289.3, 433.3],
                       [293.7, 433.3],
                       [293.7, 434.7],
                       [292.3,434.7],
                       [292.3,433.7],
                       [290.7,433.7],
                       [290.7,435.3],
                       [291.7,435.3],
                       [291.7,436.3],
                       [293.7,436.3],
                       [293.7,443.3],
                       [294.7,443.3],
                       [294.7,443.7],
                       [288.3,443.7],
                       [288.3,443.3],
                       [289.3,443.3],
                       [289.3,433.3]])
    agent_pos_goals = []
    while len(agent_pos_goals) != nb_points:
        random_goal = np.random.random((1,3)) * 2 - 1
        g_x = unscale_vector(random_goal[0,0],np.array(b.get_bounds(['agent_x'])))
        g_z = unscale_vector(random_goal[0,2],np.array(b.get_bounds(['agent_z'])))
        if inside_polygon(g_x, g_z, points):
            agent_pos_goals.append(random_goal)
    return np.array(agent_pos_goals).reshape(nb_points,-1)
# generate random goals for evaluation
np.random.seed(42) #always use same random goals
agent_pos_goals = generate_goals_in_arena(850)
np.random.seed(43)
cart_x_goals = np.random.random((100,1)) * 2 - 1
np.random.seed(44)
breads_goals = np.random.random((50,1)) * 2 - 1

with open('large_final_test_set_goals.pickle', 'wb') as handle:
    pickle.dump([agent_pos_goals, cart_x_goals, breads_goals],
    			handle,
    			protocol=pickle.HIGHEST_PROTOCOL)