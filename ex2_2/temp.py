import numpy as np
import matplotlib.pyplot as plt
import math

ac = np.zeros(100000)
mc = np.zeros(100000)
mnc = np.zeros(100000)

for i in range(100000):
    ac[i] = 0.01* (np.random.normal()+5)*i/100000
    mnc[i] = 0.01 * np.random.normal()/10000
for i in range(30000):
    #print(math.exp(i))
    mc[i] = np.log(i)/10 + 0.01 * np.random.normal() - 0.13
for j in range(70000):
    mc[j+30000] = 0.01 * np.random.normal()+0.9
plt.plot(ac)
plt.plot(mc)
plt.plot(mnc)
plt.show()
