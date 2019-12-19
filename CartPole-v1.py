import argparse
import sys
import random
import torch
import torch.nn.functional as F

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt


class Qnn(torch.nn.Module):
    def __init__(self, e_size, a_size):
        super(Qnn, self).__init__()
        self.c1 = torch.nn.Linear(e_size, 10)
        self.c2 = torch.nn.Linear(10, a_size)
    def forward(self, e):
        x = self.c1(e)
        x = F.relu(x)
        x = self.c2(x)
        return torch.sigmoid(x)

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
        n_action = env.action_space.n
        n_input = env.observation_space.shape[0]
        # NEURAL NETWORK
        self.net = Qnn(n_input, n_action)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.00001)
        self.buff = Buffer(buffer_size)
        self.reward_sum = 0
        self.env = env
        self.tau = 0.4
        self.gamma = 0.9
    def act(self, Q):
        tirage = random.random()
        sum_ = 0
        den_ = torch.exp(Q / self.tau).sum()
        for i in range(Q.shape[0]):
            sum_ += torch.exp(Q[i]/self.tau) / den_
            if(sum_>=tirage):
                break
        return i

    def reset(self):
        self.reward_sum = 0
        self.ob = self.env.reset()

    def step(self):
        etat_ = self.ob
        x = torch.Tensor(etat_)
        y = self.net.forward(x)
        
        action = self.act(y)
        self.ob, reward, self.done, _ = self.env.step(action)

        self.reward_sum += reward

        inter = Interaction(etat_, action, self.ob, reward, self.done)
        self.buff.append(inter)


    def learn(self):
        batch = self.buff.sample(10)
        for exp in batch:
            if(exp.f):
                e = torch.Tensor(exp.e)
                s = torch.Tensor(exp.s)
                Q = self.net.forward(e)[exp.a]
                Qc = self.net.forward(s)
                loss = (Q - (exp.r + self.gamma * Qc.max())) ** 2
                loss.backward()
                self.optimizer.step()
            else:
                e = torch.Tensor(exp.e)
                Q = self.net.forward(e)[exp.a]
                loss = (Q - exp.r) ** 2
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

    agent2 = DQN_Agent(env, 100000)

    episode_count = 100
    epoch = 10

    reward = 0
    done = False
    
    reward_sums = []

    fig_rewards = plt.figure()
    for i in range(episode_count):
        agent2.reset()
        while True:
            agent2.step()
            agent2.learn()
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
