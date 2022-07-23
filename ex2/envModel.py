import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class EnvModel():
    def __init__(self, actions, range=2, simThr=0.9):

        #self.StateModel = pd.DataFrame(columns=actions, dtype=np.float64)
        self.StateModel = pd.DataFrame(columns=[0], dtype=np.float64)
        self.transModel = pd.DataFrame(columns=actions, dtype=np.float64)
        #print('1', self.StateModel, '2', self.transModel)
        #self.decayModel = pd.DataFrame(columns=actions, dtype=np.float64)
        self.event = np.zeros((1, 2))
        self.range = range
        self.simThr = simThr
        self.actions = actions
        self.epi = 1

    def stateCompute(self, data):
        temp = data
        
        temp[0] = int(temp[0]/0.2)
        temp[1] = int(temp[1]/0.2)
        flag = 0
        for i in range(self.event.shape[0]):
            if self.event[i, 0] == temp[0] and self.event[i, 1] == temp[1]:
                #return i 
                flag = i
        if flag==0:
            self.event = np.vstack((self.event, temp))
            return self.event.shape[0]
        else:
            return flag


    def modelLearn(self, action, state, state_, reward, done):

        self.checkStateExist(state)
        self.checkStateExist(state_)        
        if not done:
            #if state != state_:
            #state trans model learn
            self.transModel.loc[state, action] = state_
            #reward learn
            self.StateModel.loc[state, 0] = reward
            #decay learn
            #self.decayModel.loc[state, action] = 1
        else:
            self.transModel.loc[state, action] = state_
            self.StateModel.loc[state, 0] = reward
            #self.decayModel.loc[state, action] = 1
        #print('state', state, 'state_', state_, 'reward',self.rewardModel.loc[state, action], 'done',done)
        #self.mapShow()
    
    def checkStateExist(self, state):
        if state not in self.StateModel.index:
            # append new state to q table
            #print(self.StateModel)
            self.transModel = self.transModel.append(pd.Series([0]*len(self.actions), index=self.transModel.columns, name=state,))
            self.StateModel = self.StateModel.append(pd.Series(0, index=self.StateModel.columns, name=state,))
            #self.decayModel = self.decayModel.append(pd.Series([0]*len(self.actions), index=self.StateModel.columns, name=state,))
            
    
    def saveModel(self):
        print('save model')
        self.StateModel.to_csv('/home/hzy/Desktop/ex2/dataset/StateModel.csv')
        self.transModel.to_csv('/home/hzy/Desktop/ex2/dataset/transModel.csv')
        #self.decayModel.to_csv('/home/zac/zac/Ypaper3/ex2/dataset/decayModel.csv')
        np.savetxt('/home/hzy/Desktop/ex2/dataset/event', self.event)

    def mapShow(self):
        x = self.event[:, 0]*(2*self.range)-self.range
        y = self.event[:, 1]*(2*self.range)-self.range
        fig1 = plt.figure('postion')
        plt.scatter(x, y, s=5, c='red')
        #plt.xlim(-2, 2)
        #plt.ylim(-2, 2)
        plt.pause(0.01)
        fig1.clf(0.001)
    
    def chooseAction(self, state):
        state = int(state)
        self.checkStateExist(state)
        #print(self.rewardModel.loc[state, :])
        stateAction = self.transModel.loc[state, :]
        actions = stateAction[stateAction == 0]
        
        if len(actions)>0:
            action = np.random.choice(stateAction[stateAction == 0].index)
            #print(action)
            return action, 0
        else:
            action = np.random.choice([0, 3])
            return action, 0
        '''
        if np.random.uniform(0, 1)<self.epi:
            actions = self.transModel.loc[state, :]
            print('actions', actions, 'action', actions[actions==np.max(actions)].index)
            action = np.random.choice(actions[actions==np.max(actions)].index)
            #print(action)
            return action, np.max(actions)
        else:
            action = np.random.choice(self.actions)
            return action, 0
        '''
        
    def modelUpdate(self):
        return self.StateModel, self.transModel, self.event