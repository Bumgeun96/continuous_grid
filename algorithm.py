# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sac_network import SoftQNetwork,Actor
from replay_memory import ReplayMemory as memory


class SAC_agent():
    def __init__(self,environment,args):
        # Initialize with args
        self.seed = args.seed
        self.buffer_size = args.buffer_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.learning_start = args.learning_start
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.policy_frequency = args.policy_frequency
        self.target_network_frequency = args.target_network_frequency
        self.noise_clip = args.noise_clip
        self.alpha = args.alpha
        self.auto_tune = args.auto_tune
        self.global_step = 0
        
        # Set a random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        
        self.replay_memory = memory(self.buffer_size,self.batch_size,self.seed)
        self.visit = defaultdict(lambda: np.zeros(1))
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(environment).to(self.device)
        self.critic1 = SoftQNetwork(environment).to(self.device)
        self.critic2 = SoftQNetwork(environment).to(self.device)
        self.critic_target1 = SoftQNetwork(environment).to(self.device)
        self.critic_target2 = SoftQNetwork(environment).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())
        self.critic_target2.load_state_dict(self.critic2.state_dict())
        self.critic_optimizer = optim.Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=self.critic_lr)
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=self.actor_lr)
        
        # if self.auto_tune:
        #     target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        #     log_alpha = torch.zeros(1, requires_grad=True, device=device)
        #     alpha = log_alpha.exp().item()
        #     a_optimizer = optim.Adam([log_alpha], lr=self.critic_lr)
        # else:
        #     alpha = self.alpha
            
            
    def action(self,state):
        if self.global_step < self.learning_start:
            action = torch.tensor([np.random.uniform(-1, 1),np.random.uniform(-1, 1)])
        else:
            state = state.to(self.device)
            action, _, _ = self.actor.get_action(state)
        self.global_step += 1
        return action
    
    def store_experience(self,state,action,reward,next_state,terminal):
        self.replay_memory.add(state,action,reward,next_state,terminal)
        
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
        
    def training(self):
        if self.global_step > self.learning_start:
            experiences = self.replay_memory.sample()
            states, actions, rewards, next_states, terminations = experiences
            states = torch.tensor(states).to(self.device)
            actions = torch.tensor(actions).to(self.device)
            rewards = torch.tensor(rewards).to(self.device)
            next_states = torch.tensor(next_states).to(self.device)
            terminations = torch.tensor(terminations).to(self.device)
            with torch.no_grad():
                next_actions, next_state_log_pi, _ = self.actor.get_action(next_states)
                q1_target = self.critic_target1(next_states,next_actions)
                q2_target = self.critic_target2(next_states,next_actions)
                next_Q = torch.min(q1_target,q2_target) - self.alpha*next_state_log_pi
                target = rewards+(1-terminations)*self.gamma*next_Q
                
            q1 = self.critic1(states,actions)
            q2 = self.critic2(states,actions)
            q1_loss = F.mse_loss(q1,target)
            q2_loss = F.mse_loss(q2,target)
            q_loss = q1_loss + q2_loss
            
            self.critic_optimizer.zero_grad()
            q_loss.backward()
            self.critic_optimizer.step()
            
            # TD3 delayed update
            if self.global_step % self.policy_frequency == 0:
                for _ in range(self.policy_frequency):
                    pi, log_pi, _ = self.actor.get_action(states)
                    q1_pi = self.critic1(states,pi)
                    q2_pi = self.critic2(states,pi)
                    q_pi = torch.min(q1_pi,q2_pi).view(-1)
                    actor_loss = ((self.alpha*log_pi)-q_pi).mean()
                    
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    
                    # if self.auto_tune:
                    #     with torch.no_grad():
                    #         _,log_pi,_ = self.actor.get_action(states)
                    #     alpha_loss = (-log_alpha*(log_pi+target_entropy)).mean()
                        
                    #     a_optimizer.zero_grad()
                    #     alpha_loss.backward()
                    #     a_optimizer.step()
                    #     alpha = log_alpha.exp().item()
            
            if self.global_step % self.target_network_frequency == 0:
                self.soft_update(self.critic1,self.critic_target1)
                self.soft_update(self.critic2,self.critic_target2)
                
    def count_visiting(self,state):
        self.visit[(int(np.around(state)[0]),int(np.around(state)[1]))] += 1
    
    def get_visiting_time(self):
        visit_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                visit_table[row][col] = self.visit[(row,col)]
        return visit_table
            
            
            
    




import numpy as np
from collections import defaultdict
import random

class Q_learning():
    def __init__(self, environment, epsilon=0.05, alpha=0.01, gamma=0.99):
        self.Q_table = defaultdict(lambda: np.zeros(4)) # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.visit = defaultdict(lambda: np.zeros(1))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = environment.actions # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max
        
    def get_Q_table(self):
        return self.Q_table
    
    def count_visiting(self,state):
        self.visit[state] += 1

    def action(self, state):
        
        '''
        In this code, you have to implement the behavior policy (epsilon-greedy policy) w.r.t. the Q-table.
        The policy takes a state and then samples an action among  ['UP', 'DOWN', 'LEFT', 'RIGHT'],
        and you can index the above actions as [0, 1, 2, 3]. Use "self.epsilon" and "self.Q_table".
        '''
        action_dim  = len(self.actions)
        argmax_a_prop = self.epsilon/action_dim+1-self.epsilon
        a_prop = self.epsilon/action_dim
        argmax_a_idx = np.argmax(self.Q_table[state])
        
        #assign probability to each action.
        policy = [a_prop]*action_dim
        policy[argmax_a_idx] = argmax_a_prop
        
        #Epsilon greedy policy
        action_index = self.actions.index(random.choices(self.actions,policy,k=1)[0])

        return self.actions[action_index]

    def update(self, current_state, next_state, action, reward):
        '''
        In this code, you should implement Q-learning update rule.
        '''
        done = False
        if next_state == (0,9):
            done = True
        action_idx = self.actions.index(action)
        self.Q_table[current_state][action_idx] = self.Q_table[current_state][action_idx]\
            +self.alpha*(reward+self.gamma*self.Q_table[next_state].max()*(1-int(done))-self.Q_table[current_state][action_idx])

    def get_max_Q_function(self):
        '''
        This code gives max_a Q(s,a) for each state to us. The output of this code should be a form of "list".
        Therefore, the output "max_Q_table = [max_a Q(s,a)] = [max_a Q((row_index, col_index),a)]",
         and you already found the index of state "s" in GridWorld.py.
        '''
        max_Q_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                max_Q_table[row][col] = self.Q_table[(row,col)].max()
        return max_Q_table
    
    def get_visiting_time(self):
        visit_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                visit_table[row][col] = self.visit[(row,col)]
        return visit_table

class Double_Q_learning():
    def __init__(self, environment, epsilon=0.05, alpha=0.01, gamma=0.99):
        self.Q1 = defaultdict(lambda: np.zeros(4)) # Q_table = {"state":(Q(s,UP),Q(s,DOWN),Q(s,LEFT),Q(s,RIGHT),)}
        self.Q2 = defaultdict(lambda: np.zeros(4))
        self.Q_table = defaultdict(lambda: np.zeros(4))
        self.visit = defaultdict(lambda: np.zeros(1))
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = environment.actions # ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.env_row_max = environment.row_max
        self.env_col_max = environment.col_max
        
    def get_Q_table(self):
        return self.Q_table
    
    def count_visiting(self,state):
        self.visit[state] += 1

    def action(self, state):
        
        action_dim  = len(self.actions)
        argmax_a_prop = self.epsilon/action_dim+1-self.epsilon
        a_prop = self.epsilon/action_dim
        self.Q_table[state] = (self.Q1[state]+self.Q2[state])/2
        argmax_a_idx = np.argmax(self.Q_table[state])
        
        #policy
        policy = [a_prop]*action_dim
        policy[argmax_a_idx] = argmax_a_prop
        
        #Epsilon greedy policy
        action_index = self.actions.index(random.choices(self.actions,policy,k=1)[0])
        return self.actions[action_index]

    def update(self, current_state, next_state, action, reward):
        done = False
        if next_state == (0,9):
            done = True
        action_idx = self.actions.index(action)
        if random.uniform(0,1)<0.5:
            a_prime = np.argmax(self.Q1[next_state])
            self.Q1[current_state][action_idx] = self.Q1[current_state][action_idx]\
                + self.alpha*(reward+self.gamma*self.Q2[next_state][a_prime]*(1-int(done))-self.Q1[current_state][action_idx])
        else:
            a_prime = np.argmax(self.Q2[next_state])
            self.Q2[current_state][action_idx] = self.Q2[current_state][action_idx]\
                + self.alpha*(reward+self.gamma*self.Q1[next_state][a_prime]*(1-int(done))-self.Q2[current_state][action_idx])


    def get_max_Q_function(self):
        max_Q_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                max_Q_table[row][col] = ((self.Q1[(row,col)]+self.Q2[(row,col)])/2).max()

        return max_Q_table
    
    def get_visiting_time(self):
        visit_table = np.zeros((self.env_row_max, self.env_col_max))
        for row in range(self.env_row_max):
            for col in range(self.env_col_max):
                visit_table[row][col] = self.visit[(row,col)]
        return visit_table