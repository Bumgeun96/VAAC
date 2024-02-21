import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import ListedColormap

def visualization(env,Q,rnd,entropy,v_data):
    row_max = env.row_max
    col_max = env.col_max
    fig, ax = plt.subplots(2, 2, figsize=(20, 16))
    
    
    ax[0][0].set_xlim((-0.5, col_max - 0.5))
    ax[0][0].set_ylim((row_max - 0.5, -0.5))
    im = ax[0][0].imshow(Q)
    fig.colorbar(im, ax=ax[0][0], shrink=1)
    ax[0][0].set_title('Q')
    
    
    ax[0][1].set_xlim((-0.5, col_max - 0.5))
    ax[0][1].set_ylim((row_max - 0.5, -0.5))
    im = ax[0][1].imshow(rnd)
    fig.colorbar(im, ax=ax[0][1], shrink=1)
    ax[0][1].set_title('RND')
    
    
    ax[1][0].set_xlim((-0.5, col_max - 0.5))
    ax[1][0].set_ylim((row_max - 0.5, -0.5))
    im = ax[1][0].imshow(entropy)
    fig.colorbar(im, ax=ax[1][0], shrink=1)
    ax[1][0].set_title('Entropy')
    
    
    ax[1][1].set_xlim((-0.5, col_max - 0.5))
    ax[1][1].set_ylim((row_max - 0.5, -0.5))
    im = ax[1][1].imshow(v_data,vmax = 10,cmap='hot')
    fig.colorbar(im, ax=ax[1][1], shrink=1)
    ax[1][1].set_title('Visitation')
    
    walls = np.zeros([row_max, col_max])
    for w in env.wall:
        if w != (0, env.col_max - 1):
            walls[w] = 1
        else:
            walls[w] = None
    im = ax[1][1].imshow(walls,cmap='Blues',alpha=0.5)
    
    fig.savefig("RND.pdf")

def plot_visiting(ax,fig,env,visiting_time):
    row_max = env.row_max
    col_max = env.col_max
    ax.set_xlim((-0.5, col_max - 0.5))
    ax.set_ylim((row_max - 0.5, -0.5))
    im = ax.imshow(visiting_time, vmax = 10,cmap='hot')
    fig.colorbar(im, ax=ax, shrink=1)
    
    walls = np.zeros([row_max, col_max])
    for w in env.wall:
        if w != (0, env.col_max - 1):
            walls[w] = 1
    ax.imshow(walls,cmap='Blues',alpha=0.5)
    
def draw_env(env, savefig=True):
    plt.figure(figsize=(env.row_max, env.col_max))
    if env.map == 1:
        plt.title('Continuous 4-Room Maze', fontsize=200)
    elif env.map == 2:
        plt.title('Continuous 16-Room Maze', fontsize=200)

    # Placing the initial state on a grid for illustration
    initials = np.zeros([env.row_max, env.col_max])
    initials[env.initial_location[0], env.initial_location[1]] = 1
    
    # Placing the wall on a grid for illustration
    walls = np.zeros([env.row_max, env.col_max])
    for w in env.wall:
        if w != (0, env.col_max - 1):
            walls[w] = 4

    # Make a discrete color bar with labels
    colors = {0: '#F9FFA4', 1: '#B4FF9F', 2: '#000000'}
    cm = ListedColormap([colors[x] for x in colors.keys()])
    plt.imshow(initials + walls, cmap=cm)
    plt.tick_params(axis='both', labelsize=150)
    plt.xlim((-0.5, env.col_max - 0.5))
    plt.ylim((env.row_max - 0.5, -0.5))
    plt.xticks([0, 20, 40, 60, 80, 100], [0, 20, 40, 60, 80, 100])
    plt.yticks([0, 20, 40, 60, 80, 100], [100, 80, 60, 40, 20, 0])

    if savefig:
        if env.map == 1:
            plt.savefig('./map_image/Continuous4RoomMaze.png')
        elif env.map == 2:
            plt.savefig('./map_image/Continuous16RoomMaze.png')
        
def save_pickle(data,name):
    with open("./pickle_file/"+name+".pickle","wb") as fw:
        pickle.dump(data,fw)
        print('saved')
