import numpy as np
import tensorflow as tf
import random
from collections import deque
class ActorCritic():
    def __init__(self):
        self.gamma = 0.95
        self.learn_rate = 0.001
        self.epi = 0.01
        self.state_dim = 2
        self.action_dim = 4
        self.criticModel, self.actorModel = self.createNetwork()
        self.replay_size = 2000
        self.replay_buffer = deque(maxlen=self.replay_size)

    def createNetwork(self):
        criticModel = tf.keras.Sequential()
        criticModel.add(tf.keras.layers.Dense(128, input_dim=self.state_dim, activation='relu'))
        criticModel.add(tf.keras.layers.Dense(1, activation='linear'))
        criticModel.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=self.learn_rate))

        actorModel = tf.keras.Sequential()
        actorModel.add(tf.keras.layers.Dense(20, input_dim=self.state_dim, activation='relu'))
        actorModel.add(tf.keras.layers.Dense(self.action_dim, activation='softmax'))
        actorModel.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=self.learn_rate))

        return criticModel, actorModel

    def chooseAction(self, state):
        actions = self.actorModel.predict(np.array([state]))
        actions_ = actions[0]
        #print(actions_)
        #action = np.random.choice(range(actions.shape[1]), p=actions.ravel())
        if(np.random.uniform()<self.gamma):
            action = np.argsort(-actions_)[0]
            #print(action)
            return action
        else:
            action = np.random.choice([0, 1, 2, 3])
            return action

    def saveModel(self):
        print('save model')
        self.criticModel.save('Critic.h5')
        self.actorModel.save('actor.h5')

    def remember(self, state, action, reward, nstate):
        self.replay_buffer.append((state, action, reward, nstate))

    def train(self, s, a):
        if len(self.replay_buffer) < self.replay_size:
            return
        replay_batch = random.sample(self.replay_buffer, 64)
        s_batch = np.array([replay[0] for replay in replay_batch])
        next_s_batch = np.array([replay[3] for replay in replay_batch])

        Q = self.criticModel.predict(s_batch)
        next_Q = self.criticModel.predict(next_s_batch)
        for i,replay in enumerate(replay_batch):
            _,_,r,_ = replay
            Q[i] = r + self.gamma * next_Q[i]
        self.criticModel.fit(s_batch,Q,verbose = 0)

        s = np.array([s])
        a = np.array([a])
        Q_now = self.criticModel.predict(s)
        #print(Q_now.ravel())
        self.actorModel.fit(s, a, sample_weight=Q_now.ravel(), verbose = 0)


