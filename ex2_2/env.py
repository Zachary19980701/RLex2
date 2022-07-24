from turtle import distance
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Tuple
import sympy as sy
#from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Env():
    def __init__(self, actions, epi, goalX=5, goalY=5):
        #init params of goal and actions
        self.actions = actions
        self.goalX = goalX
        self.goalY = goalY
        self.x = 0
        self.y = 0
        self.startYaw = 0
        self.yaw = self.startYaw
        self.ob_ = np.zeros(2)
        #self.sift = cv2.SIFT_create()
        self.thr = 0.2
        #init ros topoic
        #self.pubCmdVel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.angle = 0
        self._Ywas = np.linspace(-math.pi, math.pi, len(self.actions), endpoint=False)
        self.epi = epi
        

    def reset(self):
        self.x = 0
        self.y = 0
        self.yaw = self.startYaw
        ob = np.zeros(2)
        ob[0] = self.x
        ob[1] = self.y
        return ob

    
    

    def observation(self):
        self.ob_[0] = self.x
        self.ob_[1] = self.y

    def collision_occurred(self):
        if self.x<-5 or self.x>5 or self.y<-5 or self.y>5:
            return True
        else:
            return False
        '''
        elif self.x>2 and self.y>6:
            return True
        elif self.x<-6 and self.y>6:
            return True
        elif self.x>6 and self.y<-6:
            return True
        elif self.x<-6 and self.y<-6:
            return True
        '''

    def goal_occurred(self):
        #goal = sy.Point(self.goalX, self.goalY)
        #position = sy.Point(self.x, self.y)
        distance = math.sqrt((self.x - self.goalX)**2 + (self.y - self.goalY)**2)
        #print('distance', distance)
        if(distance < self.thr):
            return True, distance
        else:
            return False, distance
    
    def getReward(self):
        reward = 0
        done = False
        laser = self.collision_occurred()
        goalFlag, goalDistance = self.goal_occurred()
        #print('position', self.ob_[0], self.ob_[1], laser)
        if(goalFlag):
            if np.random.uniform(0, 1)<self.epi:
                reward = 1000
                done = True
            else:
                reward = 0
                done = True
        elif(laser):
            reward = -30
            done = True
        else:
            #print(self.ob_[0], self.ob_[1])
            reward = 0
            done = False
        return reward, done
    
    def doAction(self, action, distance=float(0.1)):
        #_Ywas = np.linspace(-math.pi, math.pi, len(self.actions), endpoint=False)
        #print(self._Ywas[action], action)
        #print("-------------------------------------------",)
        self.yaw = self._Ywas[action]
        #print(self.yaw)
        if self.yaw == 0:
            self.y += distance
        elif abs(self.yaw) == math.pi:
            self.y -= distance
        else:
            slope = math.tan(math.pi / 2 - self.yaw)
            starting_x = self.x
            if self.yaw < 0:
                self.x -= math.sqrt(
                    distance ** 2 / (slope ** 2 + 1))
            else:
                self.x += math.sqrt(
                    distance ** 2 / (slope ** 2 + 1))
            self.y -= (
                    slope * (starting_x - self.x))
        #self.render()
    
    def step(self, action):
        self.doAction(action)
        self.observation()
        reward, done = self.getReward()
        #self.render()
        return reward, done, self.ob_
        
    def render(self):
        tipX = [6, 6, 6, 6, 6, 6, -6, -6, -6, -6, -6, -6, 6]
        tipY = [6, 6, 6, -6, -6, -6, -6, -6, -6, 6, 6, 6, 6]
        fig3 = plt.figure('map')
        plt.plot(tipX, tipY)
        plt.scatter(self.goalX, self.goalY, c='red', s=100)        
        plt.scatter(self.x, self.y)
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        plt.pause(0.01)
        fig3.clf()