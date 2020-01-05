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
        return x

class DQN_Agent(object):
    def __init__(self, env, params, net=None, reward=[], loss=[]):
        self.params = params
        self.r_sums = reward
        self.l_means = loss
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
        self.criterion = params["criterion"]()
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
    # on remet la récompense cumulée à 0 et on reset l'env, etc...
    def reset(self):
        self.reward_sum = 0
        self.loss_sum = 0
        self.cpt_app = 0
        self.ob = self.env.reset()
        self.done = False
        self.action = self.env.action_space.sample()
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
        y = self.net(x).flatten()
        # Récupération de l'action selon la stratégie d'exploration
        self.action = self.act(y)

        self.ob, reward, self.done, _ = self.env.step(self.action)
        s = torch.Tensor(self.ob)
        # On sauvegarde l'interaction dans la RAM même si cuda est activé
        inter = Interaction(e.cpu(), self.action, s.cpu(), reward, self.done)
        self.buff.append(inter)
        self.reward_sum += reward

        if self.cuda:
            torch.cuda.empty_cache()

    # Ici l'agent choisi toujours la meilleure action
    def step_test(self):
        x = torch.Tensor(self.ob).unsqueeze(0)
        y = self.net.forward(x).flatten()
        # on prend la meilleure action
        self.action = y.max(0)[1].item()
        self.ob, reward, self.done, _ = self.env.step(self.action)
        self.reward_sum += reward

    def learn(self):
        # Reset du grad
        self.optimizer.zero_grad()
        # Récupération du minibatch
        batch = self.buff.sample(self.batch_size)
        # On récupère tous les états, actions, reward,... du minibatch pour les mettre dans un tenseur
        e = torch.cat(tuple(exp.e.unsqueeze(0) for exp in batch))
        a = torch.cat(tuple(torch.Tensor([exp.a]) for exp in batch))
        s = torch.cat(tuple(exp.s.unsqueeze(0) for exp in batch))
        # Les frames sont stockées dans la RAM pour économiser la VRAM
        if self.cuda:
            e = e.cuda()
            s = s.cuda()
        r = torch.cat(tuple(torch.Tensor([exp.r]) for exp in batch))
        f = torch.cat(tuple(torch.Tensor([exp.f]) for exp in batch))

        # On calcule les Q valeurs des toutes les actions pour chaque experience du minibatch
        e = self.net.forward(e)
        # On récupère les Q valeurs des actions de l'experience
        Q = torch.index_select(e, 1, a.long()).diag()
        # On calcule les Q valeurs de toutes les actions de l'état d'arrivée
        s = self.target.forward(s)
        # On récupère les meilleures actions
        Qc = torch.max(s, 1)[0].detach()
        # Equation de Bellman
        loss = self.criterion(Q, r + (1-f) * (self.gamma * Qc))
        self.loss_sum += loss.item()
        # Rétro propagation
        loss.backward(retain_graph=False)
        self.optimizer.step()

        # Mise à jour du target network
        self.cpt_app += 1
        if(self.target_update_strategy=="freq"):
            if(self.cpt_app>self.freq_copy):
                self.cpt_app = 0
                for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                    target_param.data.copy_(param )
        elif(self.target_update_strategy=="polyak"):
            for target_param, param in zip(self.target.parameters(), self.net.parameters()):
                target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )
        else:
            raise Exception('Stratégie de mise à jour de target \'{}\' inconnue.'.format(self.target_update_strategy))

        if self.cuda:
            torch.cuda.empty_cache()

def startEpoch(agent, episode_count, training=True, save=False, save_rate=50, save_name=None):
    if save_name is None:
        save_name="auto"
    # Début de l'épisode
    for i in tqdm(range(episode_count)):
        # Sauvegarde auto
        if(save and i%save_rate==0 and i!=0 and training):
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
        # Decay de tau / epsilon
        agent.tau = agent.min_tau + (agent.tau-agent.min_tau)*agent.tau_decay
        agent.r_sums.append(agent.reward_sum)

        # Dans la phase de test la loss n'est pas importante
        if not(training):
            agent.l_means.append(-1)
        else:
            agent.l_means.append(agent.loss_sum/agent.cpt_app)
    return agent.r_sums, agent.l_means

def saveModel(agent, name):
    state = {
        "model":agent.net.state_dict(),
        "params": agent.params,
        "reward": agent.r_sums,
        "loss": agent.l_means
    }
    logger.info("Saving model : " + name +".pt")
    torch.save(state, name+'.pt')

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

    # Monitor gym pour la vidéo
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    # hyperparamètres
    EXPLO = ["greedy", "boltzmann"]
    TARGET_UPDATE = ["freq", "polyak"]
    PARAMS = {
        "gamma": 0.8,
        "max_tau": 1,
        "min_tau": 0.1,
        "tau_decay": 0.99,
        "exploration": EXPLO[0],
        "sigma": 1e-3,
        "alpha": 0.005,
        "m": 4,
        "frame_skip": 4,
        "buffer_size": 10000,
        "batch_size": 20,
        "freq_copy": 1000,
        "target_update_strategy": TARGET_UPDATE[1],
        "optimizer": torch.optim.RMSprop,
        "criterion": torch.nn.MSELoss
    }

    # Si cuda activé en paramètre
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Si on a passé un modèle en paramètre
    if args.agent!=None:
        logger.info("Loading model : " + args.agent)
        state = torch.load(args.agent)
        agent_dqn = DQN_Agent(env, state["params"], state["model"], reward=state["reward"], loss=state["loss"])
    else:
        agent_dqn = DQN_Agent(env, PARAMS)

    # Si cuda activé en paramètre
    if args.cuda:
        agent_dqn.cuda = True
        agent_dqn.net = agent_dqn.net.cuda()
        agent_dqn.target = agent_dqn.target.cuda()

    # On récupère le nombre d'épisode à lancer
    episode_learn = args.learn
    episode_test = args.test

    # Sauvegarde auto
    autosave = False
    if args.autosave != None:
        autosave = True
        if not os.path.exists('autosave'):
            os.makedirs('autosave')

    # Début d'une époque de train
    startEpoch(agent=agent_dqn, episode_count=episode_learn, training=True, save=(args.autosave!=None), save_rate=args.autosave, save_name=args.save)
    # Époque de test
    startEpoch(agent=agent_dqn, episode_count=episode_test, training=False)

    # Sauvegarde du modèle
    if args.save:
        saveModel(agent_dqn, args.save)

    # Affichage des courbes de reward et de loss
    fig, (fig_rewards, fig_loss) = plt.subplots(1,2)
    fig_rewards.set_title('Récompense cumulée par épisode', fontsize=11)
    fig_rewards.set_xlabel('Episode N°', fontsize=9)
    fig_rewards.set_ylabel('Récompense cumulée', fontsize=9)
    fig_rewards.plot(agent_dqn.r_sums)
    fig_loss.set_title('Loss moyenne par épisode', fontsize=11)
    fig_loss.set_xlabel('Episode N°', fontsize=9)
    fig_loss.set_ylabel('MSE Loss', fontsize=9)
    fig_loss.plot(agent_dqn.l_means)
    plt.show()

    # Fermeture de l'env gym
    env.close()
