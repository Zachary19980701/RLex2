from os import access
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, Input
from tensorflow.keras.optimizers import Adam
import pandas as pd
import random


#超参数设置
EPSILON = 0.01

class Actor():
    def __init__(self, preFlag, transModel, rewardModel, event, actions=4, ep=0.95, preEventsNum=10000):
        self.actions = actions
        if preFlag==None:
            self.StateModel = pd.DataFrame(columns=list(range(actions)), dtype=np.float64)
            self.rewardModel = pd.DataFrame(columns=list(range(actions)), dtype=np.float64)
        else:
            self.transModel = pd.read_csv(transModel, index_col=0)
            self.rewardModel = pd.read_csv(rewardModel, index_col=0)
        self.event = event
        self.epi = ep
        self.preEventsNum = preEventsNum
        self.actorModel = self.createActorModel()
        self.goalState = 0
        


    def updateCriticModel(self, transModel, rewardModel, event):
        self.transModel = transModel
        self.rewardModel = rewardModel
        self.StateModel = rewardModel
        #self.decayModel = decayModel
        self.event = event
        #print(self.transModel)
        #goalState = self.rewardModel.loc[self.rewardModel[0]==1000].index
        for i in range(1, self.event.shape[0]):
            reward = self.StateModel.loc[i, 0]
            #print(reward)
            if reward==1000:
                self.goalState = i
                goalPos = event[self.goalState]         
        goalPos = event[self.goalState]
        #goalPos = [-19, -19]
        #print(goalPos)
        for i in range(1, self.event.shape[0]):
            reward = self.StateModel.loc[i, 0]
            #print(reward)
            if reward==-30:
                self.StateModel.loc[i, 0] = -100
            else:
                tempState = event[i]
                #print(tempState)
                tempReward = -((goalPos[0]-tempState[0])**2 + (goalPos[1]-tempState[1])**2)/100
                #print(tempReward)
                self.StateModel.loc[i, 0] = tempReward
        '''
        for i in range(1, self.event.shape[0]):
            tempState = event[i]
            #print(tempState)
            tempReward = -((goalPos[0]-tempState[0])**2 + (goalPos[1]-tempState[1])**2)/60
            #print(tempReward)
            self.StateModel.loc[i, 0] = tempReward
        '''

        

    def createActorModel(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim = self.preEventsNum, activation = 'relu'))
        model.add(tf.keras.layers.Dense(self.actions, activation = 'relu'))
        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(lr=EPSILON))
        return model
    
    
    '''
    def rtUpdate(self, state, action, reward):
        stateVec = np.zeros(self.preEventsNum)
        stateVec[state] = 1
        stateVec = np.array([np.int64(stateVec)])
        action = np.array([np.int64(action)])
        reward = np.array([np.int64(reward)])
        self.actorModel.fit(stateVec, action, sample_weight=reward, verbose=0)
    '''
    
    
    def useModelUpdate(self, batchSzie=256):
        stateUpdateBuffer = []
        rewardUpdateBuffer = []
        actionUpdateBuffer = []
        while(len(stateUpdateBuffer)<batchSzie):
            eventNum = random.choice(self.transModel.index)
            actionNum = random.randint(0, self.actions-1)
            #print(self.transModel.columns)
            eventNumInt = int(eventNum)
            actionNum_str = int(actionNum)
            if eventNumInt==0:
                eventNumInt = 1
            #print('-------------------------------', eventNumInt, actionNum_str)
            nextState = self.transModel.loc[eventNumInt, actionNum_str]
            #print(nextState)
            if nextState!=0:
                reward = self.StateModel.loc[nextState, 0]
                eventVec = np.zeros(self.preEventsNum)
                eventVec[eventNum] = 1
                stateUpdateBuffer.append(eventVec)
                rewardUpdateBuffer.append(reward)
                actionUpdateBuffer.append(actionNum)
            #print(len(stateUpdateBuffer))
        #print(stateUpdateBuffer, actionUpdateBuffer, rewardUpdateBuffer)
        stateUpdateBuffer = np.array(stateUpdateBuffer)
        actionUpdateBuffer = np.array(actionUpdateBuffer)
        rewardUpdateBuffer = np.array(rewardUpdateBuffer)
        self.actorModel.fit(stateUpdateBuffer, actionUpdateBuffer, sample_weight=rewardUpdateBuffer, verbose=1)
        #print('total train')
        
        
    def saveModel(self):
        print('model saved')
        self.actorModel.save('/home/zac/zac/Ypaper3/ex2/dataset/actionModel.h5')
        self.StateModel.to_csv('/home/zac/zac/Ypaper3/ex2/dataset/StateModel.csv')
        self.rewardModel.to_csv('/home/zac/zac/Ypaper3/ex2/dataset/rewardModel.csv')
        #self.decayModel.to_csv('/home/zac/zac/Ypaper3/ex2/dataset/decayModel.csv')
        #np.savetxt('/home/zac/dataset/dataset10/event', self.event)

    def chooseAction(self, state):
        stateVec = np.zeros(self.preEventsNum)
        stateVec[state] = 1
        actions = self.actorModel.predict(np.array([stateVec]), verbose=0)
        #print(actions)
        action = np.argsort(-actions)
        if self.checkStateExist(int(state)):
            self.epi=-1
        else:
            nextState = self.transModel.loc[int(state), action[0, 0]]
            #print('--------------------------------------------------', state)
            if nextState==0:
                self.epi=-1
            else:
                self.epi = -self.StateModel.loc[nextState, 0]/100
        if np.random.uniform() < self.epi:
            action = random.choice([0, 1, 2, 3])
            return action
        else:
            #print(actions)
            #print(action[0, 0])
            return action[0, 0]
        
    def checkStateExist(self, state):
        if state not in self.StateModel.index:
            return True
        else:
            return False

