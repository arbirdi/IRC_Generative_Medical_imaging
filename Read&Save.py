# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 18:26:24 2023

@author: LZJ

There are some serious problems with this code.
The images should be 512 by 512
However, the program saves the images as 217 by 217
I cannot solve this problem by now

To run this code, you need to use:

pip install numpy==1.21

to install version 1.21 of numpy

You also need two folders in the same directory as the code: Data and Fig
Data stores the mat files, and Fig stores the images generated
"""

import mat73
import matplotlib.pyplot as plt
import numpy as np


PID = []
illness = []
for i in range(3):
    data_dict = mat73.loadmat(f'Data/{i+1}.mat')
    image_arr = data_dict['cjdata']['image']
    PID.append(int(data_dict['cjdata']['PID']))
    illness.append(int(data_dict['cjdata']['label'].tolist()))
    fig = plt.imshow(image_arr)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(f'Fig/{i+1}', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()

PID = np.array(PID)
illness = np.array(illness)
np.savetxt('PID_and_Illness.txt', (PID, illness))
data = np.loadtxt('PID_and_Illness.txt')
