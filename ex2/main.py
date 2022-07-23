from env2 import Env
from envModel import EnvModel
import numpy as np
from ActorCritic import Actor
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore')
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
    ob = env.reset()
    state = envModel.stateCompute(ob)
    totalReward = []
    done_flag = False
    i = 0
    while not done_flag:
        i += 1
        j = 0
        #actorCritic.useModelUpdate(batchSzie=30)
        stepReward = 0
        actionReward = 0
        while(True):
            j += 1
            #print('steps', j)
            action, actionRewardStep = envModel.chooseAction(state)
            reward, done, ob_, done_flag = env.step(action)
            state_ = envModel.stateCompute(ob_)
            while(state == state_):
                reward, done, ob_, done_flag = env.step(action)
                state_ = envModel.stateCompute(ob_)
                if done:
                    ob = env.reset()
                    break
            envModel.modelLearn(action, state, state_, reward, done)
            state = state_
            #actionReward = actionReward + actionRewardStep
            stepReward = stepReward + reward
            if done:
                ob = env.reset()
                print('episodic', i, 'reward', stepReward, 'actionReward', actionReward, 'steps', j)
                break
        totalReward.append(stepReward)
    envModel.saveModel()
    #plt.plot(totalReward)
    #plt.show()
    
    env = Env(actions, epi=1.1, barr=0)
    ob = env.reset()
    state = envModel.stateCompute(ob)
    totalReward = []
    
    rewardModel, transModel, event = envModel.modelUpdate()
    actor = Actor(None, transModel, rewardModel, event)
    actor.updateCriticModel(transModel, rewardModel, event)
    actor.useModelUpdate()
    for i in range(50000):
        
        j = 0
        stepReward = 0
        actionReward = 0
        while(True):
            j += 1
            #env.render()
            #print('steps', j)
            action = actor.chooseAction(state)
            reward, done, ob_, done_flag = env.step(action)
            state_ = envModel.stateCompute(ob_)
            while(state == state_):
                reward, done, ob_, done_flag = env.step(action)
                state_ = envModel.stateCompute(ob_)
                if done:
                    ob = env.reset()
                    break
            envModel.modelLearn(action, state, state_, reward, done)
            actor.rtUpdate(state, action, reward)
            state = state_
            actionReward = actionReward + actionRewardStep
            stepReward = stepReward + reward
            if done:
                ob = env.reset()
                print('episodic', i, 'reward', stepReward, 'actionReward', actionReward, 'steps', j)
                rewardModel, transModel, event = envModel.modelUpdate()
                actor.updateCriticModel(transModel, rewardModel, event)
                actor.useModelUpdate()
                break
        totalReward.append(stepReward)
        
    #envModel.saveModel()
    plt.plot(totalReward)
    plt.show()
    #np.savetxt('/home/zac/zac/Ypaper3/ex1_1/dataset/reward', totalReward)


if __name__=='__main__':
    main()
    
