import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_visiting(ax,fig,env,visiting_time):
    row_max = env.row_max
    col_max = env.col_max
    ax.set_xlim((-0.5, col_max - 0.5))
    ax.set_ylim((row_max - 0.5, -0.5))
    walls = 0.000*np.ones([row_max, col_max])
    for w in env.wall:
        if w != (0, env.col_max - 1):
            walls[w] = 0
    visiting_time += walls
    im = ax.imshow(visiting_time, vmax = 10,cmap='hot')
    fig.colorbar(im, ax=ax, shrink=1)
    
def draw_env(env, savefig=True):
    plt.figure(figsize=(env.row_max, env.col_max))
    plt.title('Grid World', fontsize=20)

    # Placing the initial state on a grid for illustration
    initials = np.zeros([env.row_max, env.col_max])
    initials[env.initial_location[0], env.initial_location[1]] = 1

    # Placing the trap states on a grid for illustration
    traps = np.zeros([env.row_max, env.col_max])
    for t in env.terminal_states:
        if t != (0, env.col_max - 1):
            traps[t] = 2

    # Placing the terminal state on a grid for illustration
    terminals = np.zeros([env.row_max, env.col_max])
    terminals[(0, env.col_max - 1)] = 3
    
    # Placing the wall on a grid for illustration
    walls = np.zeros([env.row_max, env.col_max])
    for w in env.wall:
        if w != (0, env.col_max - 1):
            walls[w] = 4

    # Make a discrete color bar with labels
    labels = ['States', 'Initial\nState', 'Trap\nStates', 'Terminal\nState', 'Wall\nStates']
    colors = {0: '#F9FFA4', 1: '#B4FF9F', 2: '#FFA1A1', 3: '#FFD59E', 4: '#000000'}

    cm = ListedColormap([colors[x] for x in colors.keys()])
    norm_bins = np.sort([*colors.keys()]) + 0.5
    norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
    ## Make normalizer and formatter
    norm = BoundaryNorm(norm_bins, len(labels), clip=True)
    fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

    diff = norm_bins[1:] - norm_bins[:-1]
    tickz = norm_bins[:-1] + diff / 2
    plt.imshow(initials + traps + terminals + walls, cmap=cm, norm=norm)
    plt.colorbar(format=fmt, ticks=tickz)

    plt.xlim((-0.5, env.col_max - 0.5))
    plt.ylim((env.row_max - 0.5, -0.5))
    plt.yticks(np.linspace(env.row_max - 0.5, -0.5, env.row_max + 1))
    plt.xticks(np.linspace(-0.5, env.col_max - 0.5, env.col_max + 1))
    plt.grid(color='k')

    for loc in env.terminal_states:
        plt.text(loc[1], loc[0], 'X', ha='center', va='center', fontsize=40)
    plt.text(env.initial_location[1], env.initial_location[0], 'O', ha='center', va='center', fontsize=40)

    if savefig:
        plt.savefig('./gridworld.png')