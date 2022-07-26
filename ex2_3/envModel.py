from importlib.resources import path
from multiprocessing.connection import wait
from random import uniform
from selectors import SelectorKey
from turtle import pos
from typing import Mapping
from cv2 import waitKey
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

class EnvModel():
    def __init__(self, actions, range=2, simThr=0.9):
        self.model = nx.Graph()
        self.model.add_node(0, pos=(0, 0), value=-1, confi=1)
        self.event = np.zeros((1, 2))
        self.range = range
        self.simThr = simThr
        self.actions = actions
        self.epi = 1
        self.exploreWithGoalFlag = 0
        self.prestate = 0

    def stateCompute(self, data, done_flag, done):
        temp = data
        temp[0] = int(temp[0]/0.1)
        temp[1] = int(temp[1]/0.1)
        if done_flag:
            self.model.add_node('goal', pos=(temp[0], temp[1]), value=100, confi=0.1)
        flag = False
        for i in range(self.event.shape[0]):
            if self.event[i, 0] == temp[0] and self.event[i, 1] == temp[1]:
                epi = self.model.nodes[i]['confi']
                #print(epi)
                self.model.nodes[i]['confi']=epi + 0.1
                if done:
                    self.model.nodes[i]['value']=-30
                else:
                    self.model.nodes[i]['value']=-1
                flag = True
                return i
        if not flag:
            self.event = np.vstack((self.event, temp))
            if done:
                self.model.add_node(self.event.shape[0]-1, pos=(temp[0], temp[1]), value=-30, confi=0.1)
            else:
                self.model.add_node(self.event.shape[0]-1, pos=(temp[0], temp[1]), value=-1, confi=0.1)
            return self.event.shape[0]-1
        else:
            return 0


    def modelLearn(self, action, state, state_, reward, done, done_flag):
        if done_flag:
            self.model.add_edge(state, 'goal')
        else:
            if not done:
                self.model.add_edge(state, state_)
            else:
                if self.model.has_edge(state, state_):
                    self.model.remove_edge(state, state_)

    
    def saveModel(self):
        print('save model')
        nx.write_gpickle(self.model, 'F:/zacProject/Ypaper3/ex2_2/dataset/graph')
        np.savetxt('F:/zacProject/Ypaper3/ex2_2/dataset/event', self.event)

    def show(self):
        pos = nx.get_node_attributes(self.model, 'pos')
        #color = [self.model.get(node, 'value') for node in self.model.nodes()]
        nx.draw(self.model, pos, node_size=25)
        #plt.savefig("path.png")   
        plt.show()
        
    def chooseAction(self, state):
        if self.model.has_node('goal'):
            if nx.has_path(self.model, state, 'goal'):
                nextGoal = nx.shortest_path(self.model, state, 'goal')[1]
                action = self.computeAction(state, nextGoal)
                #print(nextGoal, action)
                if np.random.uniform()<self.model.nodes[nextGoal]['confi']:
                    return action
                else:
                    action = np.random.choice([0, 1, 2, 3])
                    return action
            else:
                self.exploreWithGoalFlag += 1
                #epis = nx.get_node_attributes()
                action = self.exploreWithGoal(state)
                #print(action)
                self.prestate = state
                return action
        else:
            action = self.exploreWihoutGoal()
            if self.exploreWithGoalFlag<3:
                    self.exploreWithGoalFlag += 1
            else:
                self.exploreWithGoalFlag = 0
            return action

    
    def exploreWihoutGoal(self):
        #explore with Breadth-first search
        action = np.random.choice([0, 3])
        return action

    def exploreWithGoal(self, state):
        #print()
        neighbors = self.model.neighbors(state)
        lastValue = -30
        neighborFLag = False
        #print()
        for neighbor in neighbors:
            neighborValue = self.model.nodes[neighbor]['value']
            if neighbor != self.prestate:
                if neighborValue>lastValue:
                    neighborFLag = neighbor
        
        if neighborFLag == False:
            action = np.random.choice([0, 3])
            return action
        else:
            #print(self.model.nodes[neighborFLag]['value'])
            action = self.computeAction(state, neighborFLag)
            if np.random.uniform()<0.6:
                return action
            else:
                action = np.random.choice([0, 3])
                return action


    def modelRecompute(self):
        if self.model.has_node('goal'):
            for node in self.model.nodes():
                #print(node)
                if self.model.nodes[node]['value']==-30:
                    self.model.nodes[node]['value']=-30
                else:
                    self.model.nodes[node]['value'] = -((self.model.nodes['goal']['pos'][0]-self.model.nodes[node]['pos'][0])**2+(self.model.nodes['goal']['pos'][1]-self.model.nodes[node]['pos'][1])**2)/3000
                #print(self.model.nodes[node]['value'])

    def computeAction(self, state, state_):
        pos0 = self.model.nodes[state]['pos']
        pos1 = self.model.nodes[state_]['pos']
        #print(pos0, pos1)
        if pos0[0]>pos1[0]:
            action = 1
        elif pos0[0]<pos1[0]:
            action = 3
        elif pos0[1]>pos1[1]:
            action = 0
        else:
            action = 2
        return action

    def modelChange(self):
        if self.exploreWithGoalFlag==1:
            nx.set_node_attributes(self.model, values=0, name='confi')

    
