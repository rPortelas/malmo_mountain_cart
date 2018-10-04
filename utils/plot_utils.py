import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats as stats
from utils.gep_utils import *


def plot_agent_pos_exploration(fig_id, Xs, Zs,cart_Xs):
    plt.close(fig_id)
    plt.figure(fig_id)

    b = Bounds()
    b.add('agent_x',[288.3,294.7])
    b.add('agent_z',[433.3,443.7])

    plt.figure(1)

    # add arena boundaries
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

    #upper floor limit
    plt.plot([293.7, 292.3], [440.7, 440.7], color='k', linestyle=':', linewidth=2)
    plt.plot([290.7, 289.3], [440.7, 440.7], color='k', linestyle=':', linewidth=2)

    #cart swinged up limit
    plt.plot([286.8, 286.8], [443, 444], color='g', linestyle='--', linewidth=2)
    plt.plot([296.2, 296.2], [443, 444], color='g', linestyle='--', linewidth=2)
    
    for i,(x,y) in enumerate(points):
        if i == (len(points) - 1):
            xs = [x,points[0,0]]
            ys = [y,points[0,1]]
        else:
            xs = [x,points[i+1,0]]
            ys = [y,points[i+1,1]]
        plt.plot(xs,ys,color='k',linestyle='--', linewidth=2)

    # add cart's target positions
    #plt.plot([296.2],[443.5],'r*',markersize=4)
    #plt.plot([286.8],[443.5],'r*',markersize=4)
    # plot (x,z) positions
    plt.plot(Xs,Zs,'r.',markersize=1)
    plt.plot(cart_Xs,[443.5]*len(cart_Xs),'bs',markersize=1)
    

    plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.axis('off')
    #plt.title("[x,z] points reached by agent at end of episode")

def plot_agent_cart_exploration(fig_id, Xs):
    plt.close(fig_id)
    plt.figure(fig_id)
    plt.hist(Xs)
    print( set(Xs))
    plt.title("Distribution of cart final pos. Was moved %s/%s" % (Xs.count(not 291.5),len(Xs)))

def plot_agent_bread_exploration(fig_id, final_bread_recovered,bread_vec):
    plt.close(fig_id)
    plt.figure(fig_id)
    plt.hist(final_bread_recovered,bins=range(0,7))
    for i in range(6):
        print( "%s bread found: %s/%s" % (i,final_bread_recovered.count(float(i)),final_bread_recovered.count(not float(i))))
    plt.title("Distribution of number of bread recovered by agent")

def plot_eval_errors(fig_id, eval_errors, cart_touched):
    plt.close(fig_id)
    plt.figure(fig_id)
    print( eval_errors[0])
    print( eval_errors[1])
    print( eval_errors[2])
    print( "cart touched %s" % sum(cart_touched))
    plt.plot(eval_errors[0], color='red', label='final agent pos error')
    plt.plot(eval_errors[1], color='blue', label='final cart pos error')
    plt.plot(eval_errors[2], color='green', label='nb breads recovered error')
    plt.legend()

def get_final_eval_error(final_errors, cart_touched, info=True):
    a_pos_err, c_pos_err, nb_bread_err = final_errors
    if info:
        print( "final evaluation errors:")
        print( "agent pos: %s" % a_pos_err)
        print( "cart pos: %s" % c_pos_err)
        print( "nb bread: %s" % nb_bread_err)
        print( "cart touched %s" % sum(cart_touched))
    return a_pos_err, c_pos_err, nb_bread_err, sum(cart_touched)


def get_grid_cell_exploration(a_Xs,a_Zs,c_Xs,nb_breads, bread_vec, c_bins=100, a_bins_x=10,a_bins_z=30, info=True):
    nb_c_bins = c_bins
    nb_a_bins_x = a_bins_x
    nb_a_bins_z = a_bins_z
    nb_a_bins = nb_a_bins_x * nb_a_bins_z
    
    a_explored_bins = len(np.unique(stats.binned_statistic_2d(a_Xs,
                                                          a_Zs,
                                                          np.arange(len(a_Xs)),
                                                          bins=nb_a_bins_x,
                                                          range=[[288.3,294.7],[433.3,443.7]]).binnumber))
    
    c_explored_bins = len(np.unique(stats.binned_statistic(c_Xs,
                                                           np.arange(len(c_Xs)),
                                                           bins=nb_c_bins,
                                                           range=[285,297]).binnumber))
    
    first_breads_found = []
    for i in range(6):
        try:
            first_found = nb_breads.index(i)
        except ValueError:
            first_found = 'never'
        nb_found = nb_breads.count(float(i))
        first_breads_found.append([first_found,nb_found])
    #print( first_breads_found

    final_bread_vec = np.zeros((len(bread_vec[0]),5))
    #reconstruct original bread vectors
    for bread_idx, v in enumerate(bread_vec):
        for iter_idx in range(len(v)):
            final_bread_vec[iter_idx,bread_idx] = v[iter_idx]
    
    b_explored_bins = len(np.unique(final_bread_vec,axis=0))

    try:
        first_cart_touched = next(x[0] for x in enumerate(c_Xs) if x[1] != 291.5)
    except StopIteration:
        first_cart_touched = 'never'
    # check for cart swing up
    try:
        first_swing_left = next(x[0] for x in enumerate(c_Xs) if x[1] > 296.2)
    except StopIteration:
        first_swing_left = 'never'
    try:
        first_swing_right = next(x[0] for x in enumerate(c_Xs) if x[1] < 286.8)
    except StopIteration:
        first_swing_right = 'never'


    if info:
        print( 'agent_pos: final cells reached: %s/%s' % (a_explored_bins,(nb_a_bins)))
        print( 'cart_pos: final cells reached: %s/%s' % (c_explored_bins,nb_c_bins))
        print( 'first time cart swinged up left: %s' % first_swing_left)
        print( 'first time cart swinged up right: %s' % first_swing_right)
        print( 'combination of recovered bread found: %s/32' % b_explored_bins)
        for i in range(6):
            print( "first time %s bread found: %s" % (i,first_breads_found[i][0]))
            print( "%s bread found: %s/%s" % (i,first_breads_found[i][1][0],first_breads_found[i][1][1]))


    return a_explored_bins, c_explored_bins, b_explored_bins,\
           first_swing_left, first_swing_right, first_breads_found, first_cart_touched


def plot_interests(fig_id,interest_dict, legend=True, labels=None):
    plt.close(fig_id)
    plt.figure(fig_id)
    colors = ['red','blue','green','magenta','black','cyan','orange']
    for i,(name,interests) in enumerate(sorted(interest_dict.items())):
        if labels is not None:
            plt.plot(interests, color=colors[i], label=labels[i],linewidth=1.5)
        else:
            plt.plot(interests, color=colors[i], label=name.replace('_',' '),linewidth=1.5)
    plt.xlabel('episodes')
    plt.ylabel('interest')
    if legend:
        leg = plt.legend(loc='upper left')
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)

def plot_goal_set(filename):
    b = Bounds()
    b.add('agent_x',[288.3,294.7])
    b.add('agent_z',[433.3,443.7])

    with open(filename, 'rb') as f:
        a_g, c_g, b_g = pickle.load(f)

    plt.figure(1)

    # add arena boundaries
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

    #upper floor limit
    plt.plot([293.7, 292.3], [440.7, 440.7], color='k', linestyle=':', linewidth=2)
    plt.plot([290.7, 289.3], [440.7, 440.7], color='k', linestyle=':', linewidth=2)
    
    for i,(x,y) in enumerate(points):
        if i == (len(points) - 1):
            xs = [x,points[0,0]]
            ys = [y,points[0,1]]
        else:
            xs = [x,points[i+1,0]]
            ys = [y,points[i+1,1]]
        plt.plot(xs,ys,color='k',linestyle='--', linewidth=2)



    g_x = unscale_vector(a_g[:,0],np.array(b.get_bounds(['agent_x'])))
    g_z = unscale_vector(a_g[:,2],np.array(b.get_bounds(['agent_z'])))
    print( len(g_x))
    plt.plot(g_x,g_z,'r.',markersize=1)
    plt.gca().invert_xaxis()
    plt.axis('equal')
    plt.title("[x,z] points reached by agent at end of episode")
    plt.show(block=False)
    


# display averaged results across multiple save files
# all files must have the same iteration number 
def plot(filename, max_it=None, show=True):
    with open(filename, 'rb') as f:
        b_k = pickle.load(f)
    print( b_k['parameters'])
    if max_it is not None:
        for k,v in b_k.items():
            if k == 'parameters':
                continue
            elif k == 'eval_errors':
                pass
            elif k == 'final_eval_errors':
                pass
            else:
                print((k))
                b_k[k] = b_k[k][:max_it]
                print(('done'))
                print((len(b_k[k])))
    a = np.array(b_k['choosen_modules'])
    unique, counts = np.unique(a, return_counts=True)
    print( dict(zip(unique, counts)))

    plot_agent_pos_exploration(1, b_k['final_agent_x_reached'], b_k['final_agent_z_reached'],b_k['final_cart_x_reached'])
    #plot_agent_cart_exploration(2, b_k['final_cart_x_reached'])
    #plot_agent_bread_exploration(3, b_k['final_bread_recovered'])
    get_grid_cell_exploration(b_k['final_agent_x_reached'],
                              b_k['final_agent_z_reached'],
                              b_k['final_cart_x_reached'],
                              b_k['final_bread_recovered'],
                              [b_k['bread_0'],b_k['bread_1'],b_k['bread_2'],b_k['bread_3'],b_k['bread_4']])
    #plot_eval_errors(2,b_k['eval_errors'])
    if 'final_eval_errors' in b_k.keys():
        get_final_eval_error(b_k['final_eval_errors'],b_k['final_eval_cart_touched'])
    if b_k['parameters']['model_type'] == "active_modular":
        plot_interests(2, b_k['interests'])
        print( b_k['parameters']['update_interest_step'])
    if show: plt.show(block=False)



