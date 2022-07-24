from importlib.resources import path
from multiprocessing.connection import wait
from cv2 import waitKey
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

class EnvModel():
    def __init__(self, actions, range=2, simThr=0.9):
        self.model = nx.Graph()
        self.model.add_node(0, pos=(0, 0))
        self.event = np.zeros((1, 2))
        self.range = range
        self.simThr = simThr
        self.actions = actions
        self.epi = 1
        self.model.add_node('goal', pos=(62, 17))

    def stateCompute(self, data):
        temp = data
        temp[0] = int(temp[0]/0.1)
        temp[1] = int(temp[1]/0.1)
        flag = False
        for i in range(self.event.shape[0]):
            if self.event[i, 0] == temp[0] and self.event[i, 1] == temp[1]:
                flag = True
                return i
        if not flag:
            self.event = np.vstack((self.event, temp))
            self.model.add_node(self.event.shape[0]-1, pos=(temp[0], temp[1]))
            return self.event.shape[0]-1
        else:
            return 0


    def modelLearn(self, action, state, state_, reward, done, done_flag):
        if done_flag:
            self.model.add_edge(state, 'goal')
        else:
            if not done:
                self.model.add_edge(state, state_, action=action)
            else:
                if self.model.has_edge(state, state_):
                    self.model.remove_edge(state, state_)

    
    def saveModel(self):
        print('save model')
        nx.write_edgelist(self.model, path)
        np.savetxt('/home/zac/zac/Ypaper3/ex2/dataset/event', self.event)

    def show(self):
        pos = nx.get_node_attributes(self.model, 'pos')
        nx.draw(self.model, pos, node_size=25)
        #plt.savefig("path.png")   
        plt.show()
        
    def chooseAction(self, state):
        if nx.has_path(self.model, state, 'goal'):
            nextGoal = nx.shortest_path(self.model, state, 'goal')[1]
            #print(self.model())
            #actions = nx.get_edge_attributes(self.model, 'action')
            #action = actions[(state, nextGoal)]
            action = self.model[state][nextGoal]['action']
            print(action)
            return action
            '''
            nextGoalPos = nx.get_node_attributes(self.model, 'pos')
            goalPos = nextGoalPos[nextGoal]
            print(goalPos)
            nowPos = nextGoalPos[state]
            if goalPos[0]-nowPos[0]>0:
                return 3
            elif goalPos[0]-nowPos[0]<0:
                return 1
            elif goalPos[1]-nowPos[1]>0:
                return 2
            else:
                return 0
            '''
        else:
            action = np.random.choice([0, 3])
            return action
