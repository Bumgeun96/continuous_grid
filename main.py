import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from Continuous_GridWorld import ContinuousGridWorld
from algorithm import SAC_agent
import argparse
from distutils.util import strtobool
from plotlib import plot_visiting,draw_env

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--n_iter_seed',type=int,default=20)
    parser.add_argument('--buffer_size',type=int,default=int(1e6))
    parser.add_argument('--gamma',type=float,default=0.99)
    parser.add_argument('--tau',type=float,default=0.005)
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--learning_start',type=int,default=1e3)
    parser.add_argument('--actor_lr',type=float,default=3e-4)
    parser.add_argument('--critic_lr',type=float,default=1e-3)
    parser.add_argument('--policy_frequency',type=int,default=2)
    parser.add_argument('--target_network_frequency',type=int,default=1)
    parser.add_argument('--noise_clip',type=float,default=0.5)
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--auto_tune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True)
    args = parser.parse_args()
    return args


def play(environment, agent, num_episodes=20, episode_length=1000, train=True):
    reward_per_episode = []
    returns = deque(maxlen=100)
    episode_return = 0

    for episode in range(num_episodes):
        timestep = 0
        terminal = False
        # while timestep < episode_length and terminal != True:
        while timestep < episode_length and terminal != True:
            with torch.no_grad():
                current_state = torch.Tensor(environment.agent_location)
                action = agent.action(current_state)
                action = np.array(action.to('cpu'))
                next_state, reward, terminal = environment.make_step(action,0)
            agent.count_visiting(next_state)
            timestep += 1

            if train:
                agent.store_experience(current_state,action,reward,next_state,terminal)
                agent.training()

            if terminal or timestep >= episode_length:
                episode_return = environment.reset()
            returns.append(reward)
        print(episode)
        
        reward_per_episode.append(np.mean(returns))
    visiting_time = agent.get_visiting_time()
    return reward_per_episode,visiting_time


def main(args):
    num_episodes = 5

    # Create environment
    env = ContinuousGridWorld()
    draw_env(env)
    
    ##################################
    v = []
    for i in range(args.n_iter_seed):
        args.seed += i
        sac_agent = SAC_agent(env,args)
        reward_per_episode,visitings = play(env,sac_agent,num_episodes,1000,True)
        v.append(visitings)
    v = np.array(v)
    sum = np.sum(v,axis=0)
    visitings = sum

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    per_name = ["Q-learning", "Double-Q-learning"]
    for i,t in enumerate([visitings]):
        plot_visiting(ax[i],fig,env,t)
        ax[i].set_title(per_name[i], size=10)
    fig.savefig("visiting_time.pdf")
    
    # Make learning curve
    plt.figure()
    plt.plot(range(1, num_episodes + 1), reward_per_episode, label="Q-learning")
    plt.xlabel("Episodes")
    plt.ylabel("Return per Episode")
    plt.legend()
    plt.savefig("learning_curve.pdf")


if __name__ == "__main__":
    args = parse_args()
    main(args)