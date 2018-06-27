import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def portrait_actor(agent, env, figure=None, definition=50, plot=True, save_figure=False, figure_file="actor.png"):
    """Portrait the actor"""
    #if env.observation_space.dim != 2:
       # raise(ValueError("The provided environment has an observation space of dimension {}, whereas it should be 2".format(env.observation_space.dim)))

    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            portrait[definition - (1 + index_y), index_x] = agent.target_actor(np.array([[x, y]]))
    if plot or save_figure:
        if figure is None:
            plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.colorbar(label="action")
        # Add a point at the center
        plt.scatter([0], [0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Actor phase portrait")
        if save_figure:
            # TODO: Create the directory if it doesn't exist
            plt.savefig(figure_file)
            plt.close()


def portrait_critic(nbsteps, agent, env, figure=None, definition=50, plot=True, save_figure=False, figure_file="critic.png"):
    #DEPRECATED, critic plot is performed in function 'plot_trajectory()'
    #if env.observation_space.dim != 2:
        #raise(ValueError("The provided environment has an observation space of dimension {}, whereas it should be 2".format(env.observation_space.dim)))

    portrait = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            action, q = agent.pi([x, y], apply_noise = False, compute_Q = False)
            portrait[definition - (1 + index_y), index_x] = 0 #agent.target_critic([x, y], action)
    if plot or save_figure:
        if figure is None:
            figure = plt.figure(figsize=(10, 10))
        plt.imshow(portrait, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
        plt.colorbar(label="critic value")
        # Add a point at the center
        plt.scatter([0], [0])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Critic phase portrait at step {}".format(nbsteps))
        if save_figure:
            # TODO: Create the directory if it doesn't exist
            plt.savefig(figure_file)
            plt.close()


def plot_trajectory(nbsteps, trajs, agent, env, figure=None, figure_file_traj="trajectory.png",
                    figure_file_critic='critic.png', definition=50, plot=True, save_figure=False
                    ):
    if figure is None:
        fig1=plt.figure(1,figsize=(10, 10))
        fig2=plt.figure(2,figsize=(10, 10))
    for i in range(len(trajs)):
        # plt.figure(1)r
        plt.scatter(trajs[i]["x"], trajs[i]["y"], c=range(1, len(trajs[i]["x"]) + 1), s=3)
    plt.colorbar(orientation="horizontal", label="steps")

    #if env.observation_space.dim != 2:
        #raise(ValueError("The provided environment has an observation space of dimension {}, whereas it should be 2".format(env.observation_space.dim)))

    # Add the actor phase portrait
    portrait_actor = np.zeros((definition, definition))
    portrait_critic = np.zeros((definition, definition))
    x_min, y_min = env.observation_space.low
    x_max, y_max = env.observation_space.high
    # Use the dimension names if given otherwise default to "x" and "y"
    x_label, y_label = getattr(env.observation_space, "names", ["x", "y"])

    for index_x, x in enumerate(np.linspace(x_min, x_max, num=definition)):
        for index_y, y in enumerate(np.linspace(y_min, y_max, num=definition)):
            # Be careful to fill the matrix in the right order
            a, q = agent.pi(np.array([x, y]), apply_noise = False, compute_Q = True)
            portrait_critic[definition - (1 + index_y), index_x] = q
            portrait_actor[definition - (1 + index_y), index_x] = a

    # TODO: Use the `corner` parameter
    plt.imshow(portrait_actor, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(label="action")
    plt.title("Actor and trajectory at step {}".format(nbsteps))
    # Add a point at the center
    plt.scatter([-np.pi/6.0], [0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if save_figure: plt.savefig(figure_file_traj)
    plt.close()

    # plt.figure(2)
    plt.imshow(portrait_critic, cmap="inferno", extent=[x_min, x_max, y_min, y_max], aspect='auto')
    plt.colorbar(label="critic value")
    # Add a point at the center
    plt.scatter([0], [0])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Critic phase portrait at step {}".format(nbsteps))
    if save_figure: plt.savefig(figure_file_critic)
    plt.close()



# def plot_distribution(actor, critic, env, actor_file="actor_distribution.png", critic_file="critic_distribution.png"):
#     """Plot the distributions of the network values"""
#     actor_actions, critic_values = network_values(env, actor, critic)
#     plot_action_distribution(actor_actions, actor_file)
#     plot_value_distribution(critic_values, critic_file)
#
#
# def plot_action_distribution(actions, file="action_ditribution.png"):
#     plt.figure(figsize=(10, 10))
#     sb.distplot(actions, kde=False)
#     plt.ylabel("probability")
#     plt.xlabel("action")
#     plt.title("Action distribution")
#     plt.savefig(file)
#     plt.close()
#
# def plot_value_distribution(values, file="value_distribution.png"):
#     plt.figure(figsize=(10, 10))
#     sb.distplot(values)
#     plt.xlabel("critic value")
#     plt.title("Value distribution")
#
#
# def action_distribution(actions, ax=None, file="action_ditribution.png"):
#     plt.figure(figsize=(10, 10))
#     sb.distplot(actions, kde=False, ax=ax)
#     plt.ylabel("probability")
#     plt.xlabel("action")
#     plt.title("Action distribution")
#     plt.savefig(file)
#     plt.close()
#
#
# def savefig(function):
#     def decorated(file, *args, **kwargs):
#         figure, ax = plt.subplots(figsize=(10, 10))
#         function(ax=ax, *args, **kwargs)
#         figure.savefig(file)
#         plt.close(figure)
#     return(decorated)
