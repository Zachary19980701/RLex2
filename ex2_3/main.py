from env2 import Env
from envModel import EnvModel
import numpy as np
import matplotlib.pyplot as plt

def main():
    actions = list(range(4))
    env = Env(actions, epi=1.1, barr=0)
    envModel = EnvModel(actions=actions)
    #envModel.show()
    done_flag = False
    done = False
    ob = env.reset()
    state = envModel.stateCompute(ob, done_flag, done)
    totalReward = []
    i = 0
    while not done_flag:
        i += 1
        j = 0
        stepReward = 0
        actionReward = 0
        ob = env.reset()
        state = envModel.stateCompute(ob, done_flag, done)
        while(True):
            #env.render()
            j += 1
            #print('steps', j)
            action = envModel.chooseAction(state)
            reward, done, ob_, done_flag = env.step(action)
            state_ = envModel.stateCompute(ob_, done_flag, done)
            envModel.modelLearn(action, state, state_, reward, done, done_flag)
            state = state_
            stepReward = stepReward + reward
            if done:
                envModel.modelRecompute()
                print('episodic', i, 'reward', stepReward, 'actionReward', actionReward, 'steps', j)
                break
        totalReward.append(stepReward)
    envModel.saveModel()
    envModel.show()
    plt.plot(totalReward)
    plt.show()
    
    env = Env(actions, epi=1.1, barr=1)
    ob = env.reset()
    state = envModel.stateCompute(ob, done_flag, done)
    totalReward = []
    done_flag = False
    i = 0
    for k in range(1000):
        i += 1
        j = 0
        stepReward = 0
        actionReward = 0
        ob = env.reset()
        state = envModel.stateCompute(ob, done_flag, done)
        while(True):
            #env.render()
            j += 1
            #print('steps', j)
            action = envModel.chooseAction(state)
            reward, done, ob_, done_flag = env.step(action)
            state_ = envModel.stateCompute(ob_, done_flag, done)
            envModel.modelLearn(action, state, state_, reward, done, done_flag)
            state = state_
            stepReward = stepReward + reward
            envModel.modelRecompute()
            if done:
                envModel.modelRecompute()
                print('episodic', i, 'reward', stepReward, 'actionReward', actionReward, 'steps', j)
                break
        totalReward.append(stepReward)
    envModel.saveModel()
    envModel.show()
        
    envModel.saveModel()
    plt.plot(totalReward)
    plt.show()
    #np.savetxt('/home/zac/zac/Ypaper3/ex1_1/dataset/reward', totalReward)


if __name__=='__main__':
    main()
    
