import argparse
import sys
import random

import gym
from gym import wrappers, logger

import matplotlib.pyplot as plt

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

    episode_count = 100
    reward = 0
    done = False
    
    reward_sums = []

    er_buffer = Buffer(1000)

    fig_rewards = plt.figure()
    for i in range(episode_count):
        ob = env.reset()
        reward_sum = 0
        while True:
            etat = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            reward_sum += reward
            inter = Interaction(etat, action, ob, reward, done)
            er_buffer.append(inter)
            print(len(er_buffer.sample(10)))
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        reward_sums.append(reward_sum)
    fig_rewards.suptitle('Récompense cumumée par épisode', fontsize=11)
    plt.xlabel('Episode N°', fontsize=9) 
    plt.ylabel('Récompense cumulée', fontsize=9)
    plt.plot(reward_sums)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
