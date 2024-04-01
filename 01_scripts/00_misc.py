# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:20:43 2023

@author: JF13832
"""

import matplotlib.pyplot as plt 
import numpy as np 
import math 
  
x = np.linspace(-10, 10, 100) 
z = 1/(1 + np.exp(-x)) 
  
plt.plot(x, z, 'k') 
plt.xlabel("x") 
plt.ylabel("Sigmoid(x)") 
for pos in ['right', 'top']: 
    plt.gca().spines[pos].set_visible(False)
    plt.legend()
plt.show() 