import gym
import numpy as np
from baselines import deepq
import pickle


BEST = 1
DEFAULT = .75
RANDOM = DEFAULT

ENV = "CartPole-v0"
FILE = "cartpole.txt"
BC_FILE = "cartpole_bc.txt"
MODEL = "cartpole_model.pkl"

s=[]
a=[]
ns=[]
d=[]

def main():
    env = gym.make(ENV)
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path="cartpole_model.pkl")
    steps = 0
    outfile = open(FILE, 'w')
    bcfile = open(BC_FILE, 'w')
    total_reward = 0
    episodes = 0

    while steps < 50000:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            #env.render()
            state_1 = obs
            s.append(state_1)

            if np.random.uniform(0,1) <= RANDOM:
                action = act(obs[None])[0]
            else:
                action = env.action_space.sample()
            
            a.append(action)

            obs, rew, done, _ = env.step(action)
            state_2 = obs
            diff = state_2 - state_1

            ns.append(state_2)
            d.append(diff)

            # write to AON file
            to_write = '['
            for w in state_1:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'

            outfile.write(to_write)
            outfile.write(" ")
            to_write = '['

            for w in state_2:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'

            outfile.write(to_write)
            outfile.write("\n")

            # write to BC file
            to_write = '['
            for w in state_1:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'

            bcfile.write(to_write)
            bcfile.write(" ")

            bcfile.write("[" + str(action) + "]")
            bcfile.write(" ")

            to_write = '['
            for w in state_2:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'

            bcfile.write(to_write)
            bcfile.write("\n")


            episode_rew += rew

            steps += 1

        print(steps)
        print("Episode reward", episode_rew)
        total_reward += episode_rew
        episodes += 1.

    print("Average reward", total_reward / episodes)
    # save state, action, nstate
    filename = 'State.npy'
    pickle.dump(s, open(filename, 'wb'))
    filename = 'Action.npy'
    pickle.dump(a, open(filename, 'wb'))
    filename = 'NState.npy'
    pickle.dump(ns, open(filename, 'wb'))
    # load state, action, nstate
    filename = 'NState.npy'
    nstate = pickle.load(open(filename, 'rb'))
    filename = 'Action.npy'
    action = pickle.load(open(filename, 'rb'))
    filename = 'State.npy'
    state = pickle.load(open(filename, 'rb'))

    outfile.close()
    bcfile.close()

if __name__ == '__main__':
    main()
