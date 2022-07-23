from turtle import distance
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Point:
    def __init__(self, x, y, yaw):
        self.x = x
        self.y = y
        self.yaw = yaw
    def plot(self):
        plt.scatter(self.x, self.y, c='red') #

class Rect:
    def __init__(self, min_x, min_y, max_x, max_y):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        
    def conlision(self, angent):
        if angent.x>self.min_x and angent.x<self.max_x and angent.y>self.min_y and angent.y<self.max_y:
            return True
        else:
            return False
    
    def plot(self):
        plt.gca().add_patch(patches.Rectangle((self.min_x, self.min_y), self.max_x-self.min_x, self.max_y-self.min_y, linewidth=1, edgecolor='r', facecolor='none'))
    
class Env():
    def __init__(self, actions, epi, barr=0):
        #init params of goal and actions
        self.actions = actions
        self.angent = Point(1.5, 4.5, 0)
        self.ob_ = np.zeros(2)
        self.thr = 0.2
        self.angle = 0
        self._Ywas = np.linspace(-math.pi, math.pi, len(self.actions), endpoint=False)
        self.epi = epi
        self.barr =barr
        self.buildRooms()
        self.buildGoal()
        

    def reset(self):
        self.angent.x = 1.5
        self.angent.y = 4.5
        self.angent.yaw = 0
        ob = np.zeros(2)
        ob[0] = self.angent.x
        ob[1] = self.angent.y
        return ob

    def buildRooms(self):
        if self.barr==0:
            self.wall0 = Rect(0, 0, 1, 6)
            self.wall1 = Rect(1, 5, 7, 6)
            self.wall2 = Rect(7, 0, 8, 6)
            self.wall3 = Rect(1, 0, 7, 1)
            self.wall4 = Rect(1, 2.5, 3, 3.5)
            self.wall5 = Rect(4, 2.5, 7, 3.5)
        else:
            self.wall0 = Rect(0, 0, 1, 6)
            self.wall1 = Rect(1, 5, 7, 6)
            self.wall2 = Rect(7, 0, 8, 6)
            self.wall3 = Rect(1, 0, 7, 1)
            self.wall4 = Rect(1, 2.5, 5, 3.5)
            self.wall5 = Rect(6, 2.5, 7, 3.5)
        
    def buildGoal(self):
        self.goal = Rect(6, 1.5, 6.5, 2)
    

    def observation(self):
        self.ob_[0] = self.angent.x
        self.ob_[1] = self.angent.y

    
    def getReward(self):
        done_flag =False
        reward = 0
        done = False
        #print('position', self.ob_[0], self.ob_[1], laser)
        if self.goal.conlision(self.angent):
            reward = 1000
            done = True
            done_flag = True
        elif self.wall0.conlision(self.angent):
            reward = -30
            done = True
        elif self.wall1.conlision(self.angent):
            reward = -30
            done = True
        elif self.wall2.conlision(self.angent):
            reward = -30
            done = True
        elif self.wall3.conlision(self.angent):
            reward = -30
            done = True
        elif self.wall4.conlision(self.angent):
            reward = -30
            done = True
        elif self.wall5.conlision(self.angent):
            reward = -30
            done = True
        else:
            reward = -0.1
            done = False
        return reward, done, done_flag
    
    def doAction(self, action, distance=float(0.2)):
        #_Ywas = np.linspace(-math.pi, math.pi, len(self.actions), endpoint=False)
        #print(self._Ywas[action], action)
        #print("-------------------------------------------",)
        self.angent.yaw = self._Ywas[action]
        #print(self.yaw)
        if self.angent.yaw == 0:
            self.angent.y += distance
        elif abs(self.angent.yaw) == math.pi:
            self.angent.y -= distance
        else:
            slope = math.tan(math.pi / 2 - self.angent.yaw)
            starting_x = self.angent.x
            if self.angent.yaw < 0:
                self.angent.x -= math.sqrt(
                    distance ** 2 / (slope ** 2 + 1))
            else:
                self.angent.x += math.sqrt(
                    distance ** 2 / (slope ** 2 + 1))
            self.angent.y -= (
                    slope * (starting_x - self.angent.x))
        #self.render()
    
    def step(self, action):
        self.doAction(action)
        self.observation()
        done_flag = False
        reward, done, done_flag = self.getReward()
        #self.render()
        return reward, done, self.ob_, done_flag
        
    def render(self):
        self.angent.plot()
        self.wall0.plot()
        self.wall1.plot()
        self.wall2.plot()
        self.wall3.plot()
        self.wall4.plot()
        self.wall5.plot()
        self.goal.plot()
        #self.wall6.plot()
        fig3 = plt.figure('map')
        plt.pause(0.01)
        fig3.clf()
        
