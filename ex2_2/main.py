from env2 import Env
from envModel import EnvModel
import numpy as np
from ActorCritic import Actor
import matplotlib.pyplot as plt

def main():
    actions = list(range(4))
    env = Env(actions, epi=1.1, barr=0)
    '''
    preTransModel = '/home/zac/dataset/dataset10/StateModel.csv'
    preEvent = '/home/zac/dataset/dataset10/event'
    preRewardModel = '/home/zac/dataset/dataset10/rewardModel.csv'
    preDecayModel = '/home/zac/dataset/dataset10/decayModel.csv'
    '''
    preTransModel = None
    preEvent = None
    preRewardModel = None
    preDecayModel = None
    envModel = EnvModel(actions=actions)
    #envModel.show()
    ob = env.reset()
    state = envModel.stateCompute(ob)
    totalReward = []
    done_flag = False
    i = 0
    while not done_flag:
        i += 1
        j = 0
        stepReward = 0
        actionReward = 0
        while(True):
            #env.render()
            j += 1
            #print('steps', j)
            action = envModel.chooseAction(state)
            reward, done, ob_, done_flag = env.step(action)
            state_ = envModel.stateCompute(ob_)
            envModel.modelLearn(action, state, state_, reward, done, done_flag)
            state = state_
            stepReward = stepReward + reward
            if done:
                ob = env.reset()
                state = envModel.stateCompute(ob)
                print('episodic', i, 'reward', stepReward, 'actionReward', actionReward, 'steps', j)
                break
        totalReward.append(stepReward)
    #envModel.saveModel()
    envModel.show()
    #plt.plot(totalReward)
    #plt.show()
    
    env = Env(actions, epi=1.1, barr=1)
    ob = env.reset()
    state = envModel.stateCompute(ob)
    totalReward = []
    done_flag = False
    i = 0
    while not done_flag:
        i += 1
        j = 0
        stepReward = 0
        actionReward = 0
        while(True):
            #env.render()
            j += 1
            #print('steps', j)
            action = envModel.chooseAction(state)
            reward, done, ob_, done_flag = env.step(action)
            state_ = envModel.stateCompute(ob_)
            envModel.modelLearn(action, state, state_, reward, done, done_flag)
            state = state_
            stepReward = stepReward + reward
            if done:
                ob = env.reset()
                state = envModel.stateCompute(ob)
                print('episodic', i, 'reward', stepReward, 'actionReward', actionReward, 'steps', j)
                break
        totalReward.append(stepReward)
    #envModel.saveModel()
    envModel.show()
        
    #envModel.saveModel()
    plt.plot(totalReward)
    plt.show()
    #np.savetxt('/home/zac/zac/Ypaper3/ex1_1/dataset/reward', totalReward)


if __name__=='__main__':
    main()
    
