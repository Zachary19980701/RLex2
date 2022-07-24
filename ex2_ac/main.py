from multiprocessing import managers
from env2 import Env
from actorCritic import ActorCritic as AC
import numpy as np
import matplotlib.pyplot as plt
def main():
    EP0 = 50000
    EP1 = 50000
    actions = [0, 1, 2, 3]
    env = Env(actions, 1)
    ac = AC()
    score_list = []
    for i in range(EP0):
        
        s = env.reset()
        score = 0
        j = 0
        while True:
            #env.render()
            j += 1
            action = ac.chooseAction(s)
            r, d, ns, _ = env.step(action)
            ac.remember(s, action, r, ns)
            ac.train(s, action)
            s = ns
            score += r
            if d:
                score_list.append(score)
                print('episode:',i,'score:',score, 'steps', j)
                #env.reset()
                break
    np.savetxt("/home/hzy/Desktop/Ypaper3/ex2_ac/ep0", score_list)
    ac.saveModel()

    env = Env(actions, 1, barr=1)
    score_list = []
    for i in range(EP1):
        #env.render()
        s = env.reset()
        score = 0
        while True:
            action = ac.chooseAction(s)
            r, d, ns, _ = env.step(action)
            ac.remember(s, action, r, ns)
            ac.train(s, action)
            s = ns
            score += r
            if d:
                score_list.append(score)
                print('episode:',i,'score:',score)
                #env.reset()
                break
    np.savetxt("/home/hzy/Desktop/Ypaper3/ex2_ac/ep1", score_list)
    ac.saveModel()

if __name__=='__main__':
    main()
