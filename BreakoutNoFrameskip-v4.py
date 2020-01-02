import argparse
import sys
import random
import torch
import torch.nn.functional as F
import copy
import time
import numpy as np
from PIL import Image

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

# Structure d'une interaction :
#   * e : état avant action
#   * a : action
#   * s : état après action
#   * r : récompense
#   * f : si interaction marque fin de l'épisode
class Interaction:
    def __init__(self,e,a,s,r,f):
        self.e = e
        self.a = a
        self.s = s
        self.r = r
        self.f = f

# Structure du buffer d'experience replay
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

    # Récupération d'un minibatch
    def sample(self, k):
        if k>len(self.buff):
            k = len(self.buff)
        return random.sample(self.buff, k)

# Agent à action random
class RandomAgent(object):
    def __init__(self, env):
        self.action_space = env.action_space
        self.env = env
        self.reward_sum = 0
        self.done = False

    def act(self):
        return self.action_space.sample()

    def step(self):
        action = agent2.act()
        ob, reward, self.done, _ = self.env.step(action)
        self.reward_sum += reward

    def reset(self):
        self.reward_sum = 0
        self.env.reset()

    def learn(self):
        return


class QModel(torch.nn.Module):
    def __init__(self, e_size, a_size):
        super(QModel, self).__init__()

        # Channel d'entrée = 1
        # Sorties de la couche = 14
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = torch.nn.Linear(3136, 512)
        self.fc2 = torch.nn.Linear(512, a_size)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        #x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.tanh(x)
        #x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        #x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x.flatten()
        return output

class DQN_Agent(object):
    def __init__(self, env, buffer_size=100000):
        self.gamma = 0.95
        self.freq_copy = 1000
        self.tau = 1
        self.tau_decay = 0.999
        self.min_tau = 0.2
        self.exploration = "greedy"
        self.sigma = 1e-3
        self.alpha = 0.01
        self.m = 4

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
        self.reward_etat = 0
        self.ob = self.env.reset()
        self.action = self.env.action_space.sample()
        self.frames = [self.preprocessing(self.ob)]
        self.etatPrec = None

    def preprocessing(self, frame):
        # Luminance relative
        etat = np.dot(frame[..., :3], [0.2126, 0.7152, 0.0722])
        # Resize
        etat = np.array(Image.fromarray(etat).resize((84,84), resample=Image.NEAREST))
        # Normalisation
        etat = etat/255
        #plt.imshow(etat, cmap = plt.get_cmap('gray'))
        #plt.show()
        return etat

    def convertFramesToInput(self, frames):
        return torch.Tensor(frames)


    # avancement de l'agent d'un pas
    def step(self, training=False):
        if(len(self.frames)<self.m):
            self.ob, reward, self.done, _ = self.env.step(self.action)
            self.reward_etat += reward
            self.frames.append(self.preprocessing(self.ob))
            if self.done:
                # Si le groupe contient un état terminal alors qu'on a pas encore m frames,
                # On duplique la frame jusqu'à en avoir le bon nombre
                self.frames.extend([self.preprocessing(self.ob)]*(self.m - len(self.frames)))
                etat_ = torch.Tensor(self.frames).unsqueeze(0)
                inter = Interaction(self.etatPrec, self.action, etat_, self.reward_etat, self.done)
                self.buff.append(inter)
        else:
            etat_ = torch.Tensor(self.frames).unsqueeze(0)
            x = etat_
            y = self.net.forward(x)
            self.action = self.act(y)
            self.ob, reward, self.done, _ = self.env.step(self.action)

            # sauvegarde de l'interraction dans le buffer
            if not(self.etatPrec is None):
                self.reward_etat += reward
                inter = Interaction(self.etatPrec, self.action, etat_, self.reward_etat, self.done)
                self.buff.append(inter)

            self.frames = [self.preprocessing(self.ob)]
            self.reward_etat = 0
            self.etatPrec = etat_

        self.reward_sum += reward


    def step_opti(self):
        etat_ = self.ob
        x = torch.Tensor(etat_)
        y = self.net.forward(x)
        action = y.max(0)[1].item()
        self.ob, reward, self.done, _ = self.env.step(action)
        self.reward_sum += reward

    def learn(self):
        batch = self.buff.sample(32)
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
                e = exp.e
                s = exp.s

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
                e = exp.e
                Q = self.net.forward(e)[exp.a]
                loss = (Q - exp.r).pow(2) 
                loss.backward()
                self.optimizer.step()
                

def startEpoch(agent, episode_count, training=True):
    r_sums = []
    for i in range(episode_count):
        agent.reset()
        while True:
            agent.step(training)
            if agent.done:
                break
        if training: agent.learn()
        r_sums.append(agent.reward_sum)
    return r_sums


if __name__ == '__main__':
    logger.set_level(logger.INFO)
    env = gym.make("BreakoutNoFrameskip-v4")

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent_dqn = DQN_Agent(env, 1000000)

    episode_learn = 500
    episode_test = 300

    # Début d'une époque de train
    reward_sums = startEpoch(agent=agent_dqn, episode_count=episode_learn, training=True)

    fig_rewards = plt.figure()
    fig_rewards.suptitle('Récompense cumumée par épisode', fontsize=11)
    plt.xlabel('Episode N°', fontsize=9) 
    plt.ylabel('Récompense cumulée', fontsize=9)
    plt.plot(reward_sums)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
