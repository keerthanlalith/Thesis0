import gym
import numpy as np
from baselines import deepq
import pickle

train = 0
test = 1
MODE = 0

BEST = 1
DEFAULT = 0.75
RANDOM = DEFAULT
num_steps = 50000

if MODE == test:
    RANDOM = BEST
    num_steps = 1000

MODEL = "Data/cartpole_model.pkl"
ENV = "CartPole-v1"

FILE = "Data/cartpole.txt"
BC_FILE = "Data/cartpole_bc.txt"
D_FILE = "Data/cartpole_d.txt"

if MODE == test:
    FILE = "Data/Tcartpole.txt"
    BC_FILE = "Data/Tcartpole_bc.txt"
    D_FILE = "Data/Tcartpole_d.txt"





def main():
    s=[]
    a=[]
    ns=[]
    d=[]
    env = gym.make(ENV)
    act = deepq.learn(env, network='mlp', total_timesteps=0, load_path=MODEL)
    steps = 0
    outfile = open(FILE, 'w')
    bcfile = open(BC_FILE, 'w')
    dfile = open(D_FILE, 'w')
    
    total_reward = 0
    episodes = 0

    while steps < num_steps:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            #env.render()
            #if steps % 200==0:
            #    obs, done = env.reset(), False
            state_1 = obs

            if np.random.uniform(0,1) <= RANDOM:
                action = act(obs[None])[0]
                obs, rew, done, _ = env.step(action)
            else:
                action = env.action_space.sample()
                obs, rew, done, _ = env.step(action)
                continue
            
            s.append(state_1)
            a.append(action)
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

            # write to Diff file
            to_write = '['
            for w in state_1:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'
            dfile.write(to_write)

            dfile.write(" ")
            
            to_write = '['
            for w in state_2:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'
            dfile.write(to_write)
            
            dfile.write(" ")
            to_write = '['

            for w in diff:
                to_write += str(w) + ','
            to_write = to_write[:-1]
            to_write += ']'

            dfile.write(to_write)

            dfile.write("\n")
 

            episode_rew += rew

            steps += 1

        print(steps)
        print("Episode reward", episode_rew)
        total_reward += episode_rew
        episodes += 1.

    outfile.close()
    bcfile.close()

    s=np.array(s)
    ns=np.array(ns)
    a=np.array(a)
    d=np.array(d)
    
    print("Average reward", total_reward / episodes)

    # save state, action, nstate
    if MODE == test:
        filename = 'Data/TState.npy'
        pickle.dump(s, open(filename, 'wb'))
        filename = 'Data/TAction.npy'
        pickle.dump(a, open(filename, 'wb'))
        filename = 'Data/TNState.npy'
        pickle.dump(ns, open(filename, 'wb'))
        filename = 'Data/TDiff.npy'
        pickle.dump(d,open(filename,'wb'))
    else: 
        # save state, action, nstate
        filename = 'Data/State.npy'
        pickle.dump(s, open(filename, 'wb'))
        filename = 'Data/Action.npy'
        pickle.dump(a, open(filename, 'wb'))
        filename = 'Data/NState.npy'
        pickle.dump(ns, open(filename, 'wb'))
        filename = 'Data/Diff.npy'
        pickle.dump(d,open(filename,'wb'))

    '''
    # load state, action, nstate
    filename = 'TNState.npy'
    nstate = pickle.load(open(filename, 'rb'))
    filename = 'TAction.npy'
    action = pickle.load(open(filename, 'rb'))
    filename = 'TState.npy'
    state = pickle.load(open(filename, 'rb'))
    filename = 'TDiff.npy'
    diff = pickle.load(open(filename, 'rb'))
    '''



if __name__ == '__main__':
    main()
