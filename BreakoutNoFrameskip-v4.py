import argparse
import sys
import random
import torch
import torch.nn.functional as F
import copy
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from collections import deque
import gym
from gym import wrappers, logger
import matplotlib.pyplot as plt
import os


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
        self.buff = deque(maxlen=self.taille)

    def append(self, inter):
        #if len(self.buff)>=self.taille:
        #    self.buff.pop(0)
        self.buff.append(inter)

    def length(self):
        return len(self.buff)

    # Récupération d'un minibatch
    def sample(self, k):
        if k>len(self.buff):
            k = len(self.buff)
        return random.sample(self.buff, k)
    def clear(self):
        self.buff.clear()

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
    def __init__(self, a_size):
        super(QModel, self).__init__()

        # Channel d'entrée = 1
        # Sorties de la couche = 14
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=3)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.bn3 = torch.nn.BatchNorm2d(64)

        #Fonction qui permet de déterminer la taille de l'entrée de la couche fully-connected
        def conv2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        fc_input_size = conv2d_size_out(conv2d_size_out(conv2d_size_out(84, 8, 4), 4, 3), 3, 1)
        fc_input_size = fc_input_size * fc_input_size*64
        #self.fc1 = torch.nn.Linear(3136, 512)
        self.fc1 = torch.nn.Linear(fc_input_size, 512)
        self.fc2 = torch.nn.Linear(512, a_size)


    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = torch.tanh(x)
        #x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = torch.tanh(x)
        #x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = torch.relu(x)
        #x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = x.flatten()
        return output

class DQN_Agent(object):
    def __init__(self, env, params, net=None):
        self.params = params
        # PARAMS
        self.gamma = params["gamma"]
        self.freq_copy = params["freq_copy"]
        self.tau = params["max_tau"]
        self.tau_decay = params["tau_decay"]
        self.min_tau = params["min_tau"]
        self.exploration = params["exploration"]
        self.sigma = params["sigma"]
        self.alpha = params["alpha"]
        self.m = params["m"]
        self.frame_skip = params["frame_skip"]
        self.target_update_strategy = params["target_update_strategy"]
        self.batch_size = params["batch_size"]
        self.cuda = False
        # NEURAL NETWORK
        self.n_action = env.action_space.n
        self.net = QModel(self.n_action)
        if net is not None:
            self.net.load_state_dict(net)
        self.target = copy.deepcopy(self.net)
        self.optimizer = params["optimizer"](self.net.parameters(), lr=self.sigma)
        self.criterion = torch.nn.MSELoss()
        self.buff = Buffer(params["buffer_size"])

        self.env = wrappers.AtariPreprocessing(env, frame_skip=self.frame_skip, screen_size=84,grayscale_obs=True, scale_obs=True) 
        self.env = wrappers.FrameStack(self.env, self.m)
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
        self.cpt_app = 0
        self.cpt_step = 0
        self.ob = self.env.reset()
        self.done = False
        self.action = self.env.action_space.sample()
        self.frames = torch.Tensor([self.preprocessing(self.ob)]*4)
        self.etatPrec = None
        if self.cuda:
            torch.cuda.empty_cache()

    def preprocessing(self, frame):
        # Luminance relative
        etat = np.dot(frame[..., :3], [0.2126, 0.7152, 0.0722])
        # Resize
        etat = np.array(Image.fromarray(etat).resize((84,84), resample=Image.NEAREST), dtype="float16")
        # Normalisation
        etat = etat/255
        #plt.imshow(etat, cmap = plt.get_cmap('gray'))
        #plt.show()
        return etat

    # avancement de l'agent d'un pas
    def step(self):
        e = torch.Tensor(self.ob)
        x = e.unsqueeze(0)
        y = self.net(x)
        self.action = self.act(y)
        self.ob, reward, self.done, _ = self.env.step(self.action)
        s = torch.Tensor(self.ob)
        inter = Interaction(e, self.action, s, reward, self.done)
        self.buff.append(inter)

        self.reward_sum += reward

        if self.cuda:
            torch.cuda.empty_cache()

        # self.cpt_step += 1
        # if(self.cpt_step%100==0):
            # self.cpt_step = 0
            # f, axarr = plt.subplots(2,2)
            # axarr[0,0].imshow(etat[0], cmap = plt.get_cmap('gray'))
            # axarr[0,1].imshow(etat[1], cmap = plt.get_cmap('gray'))
            # axarr[1,0].imshow(etat[2], cmap = plt.get_cmap('gray'))
            # axarr[1,1].imshow(etat[3], cmap = plt.get_cmap('gray'))
            # plt.show()

        # On skip un certain nombre de frames
        # if(cpt_step<self.frame_skip):
            # self.ob, reward, self.done, _ = self.env.step(self.action)
        # else:
            # cpt_step = 0

        # cpt_step += 1

        # if(len(self.frames)<self.m):
            # self.ob, reward, self.done, _ = self.env.step(self.action)
            # self.reward_etat += reward
            # self.frames.append(self.preprocessing(self.ob))
            # if self.done:
                # # Si le groupe contient un état terminal alors qu'on a pas encore m frames,
                # # On duplique la frame jusqu'à en avoir le bon nombre
                # self.frames.extend([self.preprocessing(self.ob)]*(self.m - len(self.frames)))
                # etat_ = torch.Tensor(self.frames).unsqueeze(0)
                # inter = Interaction(self.etatPrec, self.action, etat_, self.reward_etat, self.done)
                # self.buff.append(inter)
        # else:
            # etat_ = torch.Tensor(self.frames).unsqueeze(0)
            # x = etat_
            # y = self.net(x)
            # self.action = self.act(y)
            # self.ob, reward, self.done, _ = self.env.step(self.action)
            # # sauvegarde de l'interraction dans le buffer
            # if not(self.etatPrec is None):
                # self.reward_etat += reward
                # inter = Interaction(self.etatPrec, self.action, etat_, self.reward_etat, self.done)
                # self.buff.append(inter)

            # self.frames = [self.preprocessing(self.ob)]
            # self.reward_etat = 0
            # self.etatPrec = etat_

        # self.reward_sum += reward
        # if self.cuda:
            # torch.cuda.empty_cache()


    def step_test(self):
        x = torch.Tensor(self.ob).unsqueeze(0)
        y = self.net.forward(x)
        # on prend la meilleure action
        self.action = y.max(0)[1].item()
        self.ob, reward, self.done, _ = self.env.step(self.action)
        self.reward_sum += reward

    def learn(self):
        self.optimizer.zero_grad()
        batch = self.buff.sample(self.batch_size)
        e = torch.cat(tuple(exp.e.unsqueeze(0) for exp in batch))
        a = torch.cat(tuple(torch.Tensor([exp.a]) for exp in batch))
        s = torch.cat(tuple(exp.s.unsqueeze(0) for exp in batch))
        r = torch.cat(tuple(torch.Tensor([exp.r]) for exp in batch))
        f = torch.cat(tuple(torch.Tensor([exp.f]) for exp in batch))
        e = self.net.forward(e).reshape(len(batch),self.n_action)

        # On récupère les Q valeurs des actions
        Q = torch.index_select(e, 1, a.long()).diag()

        s = self.target.forward(s).reshape(len(batch),self.n_action)
        Qc = torch.max(s, 1)[0].detach()
        loss = self.criterion(Q, r + f * (self.gamma * Qc))
        loss.backward(retain_graph=False)
        self.optimizer.step()
        if(self.target_update_strategy=="freq"):
            self.cpt_app += 1
            if(self.cpt_app%self.freq_copy==0):
                self.cpt_app = 0
                for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                    target_param.data.copy_(param )
        elif(self.target_update_strategy=="polyak"):
            for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )
        else:
            raise Exception('Stratégie de mise à jour de target \'{}\' inconnue.'.format(self.target_update_strategy))

        if self.cuda:
            #print("toto")
            #print(torch.cuda.memory_allocated())
            #print(torch.cuda.memory_cached())
            torch.cuda.empty_cache()

        # for exp in batch:
            # self.optimizer.zero_grad()
            # mse = torch.nn.SmoothL1Loss()
            # if(not(exp.f)):
                # e = exp.e
                # s = exp.s

                # Q = self.net.forward(e)[exp.a]
                # Qc = self.target.forward(s)

                # loss = mse(Q, (exp.r + self.gamma * Qc.max()))
                # loss.backward()
                # self.optimizer.step()
            # else:
                # e = exp.e
                # Q = self.net.forward(e)[exp.a]
                # loss = (Q - exp.r).pow(2) 
                # loss.backward()
                # self.optimizer.step()
                

def startEpoch(agent, episode_count, training=True, save=False, save_rate=50, save_name="auto"):
    r_sums = []
    cpt_save = 0
    if save_name is None:
        save_name="auto"
    for i in tqdm(range(episode_count)):
        if(save+1 and i%save_rate==0):
            saveModel(agent, "autosave/"+save_name+"_episode"+str(i))
        agent.reset()
        while True:
            if training:
                agent.step()
                agent.learn()
            else:
                agent.step_test()
            if agent.done:
                break
        r_sums.append(agent.reward_sum)
    return r_sums

def saveModel(agent, name):
    agent.buff.clear()
    state = {
        "model":agent.net.state_dict(),
        "params": agent.params
    }
    logger.info("Saving model : " + name +".pt")
    torch.save(state, name+'.pt')
    logger.info("Model saved : " + name +".pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument("agent", nargs='?', default=None, help='Fichier d\'un agent déjà entraîné')
    parser.add_argument("--cuda", action="store_true", help="Utilise cuda pour l\'entrainement")
    parser.add_argument("--autosave", action="store", default=None, type=int, help="Active la sauvegarde auto")
    parser.add_argument("-s", "--save", action="store", help="Sauvegarde l\'agent après entraînement")
    parser.add_argument("-l", "--learn", type=int, action="store", default=500, help="Nombre d\'épisodes d\'apprentissage.")
    parser.add_argument("-t", "--test", type=int, action="store", default=0, help="Nombre d\'épisodes de test")
    args = parser.parse_args()
    logger.set_level(logger.INFO)
    env = gym.make("BreakoutNoFrameskip-v4")

    

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    EXPLO = ["greedy", "boltzmann"]
    TARGET_UPDATE = ["freq", "polyak"]
    PARAMS = {
        "gamma": 0.95,
        "max_tau": 1,
        "min_tau": 0.1,
        "tau_decay": 0.999,
        "exploration": EXPLO[0],
        "sigma": 1e-3,
        "alpha": 0.005,
        "m": 4,
        "frame_skip": 4,
        "buffer_size": 10000,
        "batch_size": 32,
        "freq_copy": 100,
        "target_update_strategy": TARGET_UPDATE[0],
        "optimizer": torch.optim.RMSprop
    }

    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    if args.agent!=None:
        logger.info("Loading model : " + args.agent +".pt")
        state = torch.load(args.agent+'.pt')
        agent_dqn = DQN_Agent(env, state["params"], state["model"])
        logger.info("Model loaded : " + args.agent +".pt")
    else:
        agent_dqn = DQN_Agent(env, PARAMS)

    if args.cuda:
        agent_dqn.cuda = True
        agent_dqn.net = agent_dqn.net.cuda()

    episode_learn = args.learn
    episode_test = args.test
    autosave = False
    if args.autosave != None:
        autosave = True
        if not os.path.exists('autosave'):
            os.makedirs('autosave')

    # Début d'une époque de train
    reward_sums = startEpoch(agent=agent_dqn, episode_count=episode_learn, training=True, save=(args.autosave!=None), save_rate=args.autosave, save_name=args.save)
    reward_sums += startEpoch(agent=agent_dqn, episode_count=episode_test, training=False)

    if args.save:
        saveModel(agent_dqn, args.save)
    fig_rewards = plt.figure()
    fig_rewards.suptitle('Récompense cumulée par épisode', fontsize=11)
    plt.xlabel('Episode N°', fontsize=9) 
    plt.ylabel('Récompense cumulée', fontsize=9)
    plt.plot(reward_sums)
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()
