import argparse
import sys
import random
import torch
import torch.nn.functional as F
import copy
import time

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt


class QModel(torch.nn.Module):
    def __init__(self, e_size, a_size):
        super(QModel, self).__init__()
        self.c1 = torch.nn.Linear(e_size, 32)
        self.c2 = torch.nn.Linear(32, 32)
        self.c3 = torch.nn.Linear(32, 32)
        self.c4 = torch.nn.Linear(32, a_size)
        # self.c4 = torch.nn.Linear(16, a_size)
        # self.c4 = torch.nn.Linear(128, a_size)
    def forward(self, e):
        x = self.c1(e)
        x = torch.relu(x)
        x = self.c2(x)
        x = torch.relu(x)
        x = self.c3(x)
        x = torch.relu(x)
        x = self.c4(x)
        # x = torch.tanh(x)
        # x = self.c4(x)
        return x

class Interaction:
    def __init__(self,e,a,s,r,f):
        self.e = e
        self.a = a
        self.s = s
        self.r = r
        self.f = f
class Buffer:
    def __init__(self, taille):
        self.taille = taille
        self.buff = []
    def append(self, inter):
        if len(self.buff)>=self.taille:
            self.buff.pop(0)
        self.buff.append(inter)
    def length(self):
        return len(self.buff)
    def sample(self, k):
        if k>len(self.buff):
            k = len(self.buff)
        return random.sample(self.buff, k)
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

class DQN_Agent(object):
    def __init__(self, env, buffer_size):
        self.gamma = 0.95
        self.freq_copy = 1000
        self.tau = 1
        self.tau_decay = 0.999
        self.min_tau = 0.2
        self.exploration = "greedy"
        self.sigma = 1e-3
        self.alpha = 0.01

        n_action = env.action_space.n
        n_input = env.observation_space.shape[0]
        # NEURAL NETWORK
        self.net = QModel(n_input, n_action)
        self.target = copy.deepcopy(self.net)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.sigma)
        # self.optimizer_target = torch.optim.SGD(self.target.parameters(), lr=0.00001)
        self.buff = Buffer(buffer_size)
        self.reward_sum = 0
        self.env = env

        self.cpt_app = 0

    # stratégie d'exploration de l'agent
    # input:
    #   * Q : liste des q-valeurs pour chaque action
    # output:
    #   * action choisie
    # exception:
    #   * si la stratégie choisie est incorrecte
    def act(self, Q):
        tirage = random.random()
        self.tau = self.min_tau + (self.tau-self.min_tau)*self.tau_decay

        if(self.exploration=="greedy"):
            if(tirage>=self.tau):
                return Q.max(0)[1].item()
            else:
                return self.env.action_space.sample()
        elif(self.exploration=="boltzmann"):
            sum_ = 0
            den_ = torch.exp(Q / self.tau).sum()
            for i in range(Q.shape[0]):
                sum_ += torch.exp(Q[i]/self.tau) / den_
                if(sum_>=tirage):
                    break
            return i
        else:
            raise Exception('Stratégie d\'exploration \'{}\' inconnue.'.format(self.exploration))

    # reset de l'agent à chaque nouvel épisode
    # on remet la récompense cumulée à 0 et on reset l'env
    def reset(self):
        self.reward_sum = 0
        self.ob = self.env.reset()

    # avancement de l'agent d'un pas
    def step(self):
        etat_ = self.ob
        x = torch.Tensor(etat_)
        y = self.net.forward(x)
        action = self.act(y)
        self.ob, reward, self.done, _ = self.env.step(action)

        self.reward_sum += reward
        
        # sauvegarde de l'interraction dans le buffer
        inter = Interaction(etat_, action, self.ob, reward, self.done)
        self.buff.append(inter)

    def step_opti(self):
        etat_ = self.ob
        x = torch.Tensor(etat_)
        y = self.net.forward(x)
        action = y.max(0)[1].item()
        self.ob, reward, self.done, _ = self.env.step(action)
        self.reward_sum += reward

    def learn(self):
        batch = self.buff.sample(20)
        for exp in batch:
            self.cpt_app += 1
            # if(self.cpt_app%self.freq_copy==0):
                # self.target.parameters.copy_(self.net.parameters())
                # for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                    # target_param.data.copy_(param )

            for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )
            self.optimizer.zero_grad()
            # self.optimizer_target.zero_grad()
            mse = torch.nn.MSELoss()
            if(not(exp.f)):
                e = torch.Tensor(exp.e)
                s = torch.Tensor(exp.s)

                Q = self.net.forward(e)[exp.a]
                Qc = self.net.forward(s)
                
                # Q2 = self.target.forward(e)[exp.a]
                Q2c = self.target.forward(s)

                loss = mse(Q, (exp.r + self.gamma * Q2c.max()))
                # loss2 = (Q2 - (exp.r + self.gamma * Qc.max())).pow(2)
                # loss = mse(Q, (exp.r + self.gamma * Qc.max()))
                # print(Q)
                # print((exp.r + self.gamma * Qc.max().item()))
                # print(loss.item())
                loss.backward()
                self.optimizer.step()
                # loss2.backward()
                # self.optimizer_target.step()
            else:
                e = torch.Tensor(exp.e)
                Q = self.net.forward(e)[exp.a]
                loss = (Q - exp.r).pow(2) 
                loss.backward()
                self.optimizer.step()
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = RandomAgent(env.action_space)

    agent2 = DQN_Agent(env, 1000000)

    episode_count = 500
    epoch = 10

    reward = 0
    done = False
    
    reward_sums = []

    fig_rewards = plt.figure()
    for i in range(episode_count):
        agent2.reset()
        while True:
            agent2.step()
            if agent2.done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        agent2.learn()
        reward_sums.append(agent2.reward_sum)
    for i in range(episode_count):
        agent2.reset()
        while True:
            agent2.step_opti()
            if agent2.done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        reward_sums.append(agent2.reward_sum)

    epochs_rec = []
    for i in range(0, len(reward_sums), epoch):
        mn = reward_sums[i:i+epoch]
        epochs_rec.append(sum(mn)/len(mn))

    fig_rewards.suptitle('Récompense cumumée par épisode', fontsize=11)
    plt.xlabel('Episode N°', fontsize=9) 
    plt.ylabel('Récompense cumulée', fontsize=9)
    plt.plot(reward_sums)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
