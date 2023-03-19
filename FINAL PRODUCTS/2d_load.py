# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 20:22:24 2023

@author: LZJ
"""

import os
import numpy as np
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
import pandas as pd

clinical_data = pd.read_csv('Clinical.csv')
id_list = clinical_data['ID']

name_list = []
grade_list = []
index_list = []
for id_name in clinical_data['ID']:
    name_list.append(id_name[-3:])
for grade in clinical_data['WHO CNS Grade']:
    grade_list.append(grade)
for i in range(len(name_list)):
    name_list[i] = '0'+name_list[i]
for i in range(len(name_list)):
    index_list.append(i)

del name_list[120]
del grade_list[120]
del index_list[-1]

del name_list[155]
del grade_list[155]
del index_list[-1]

del name_list[159]
del grade_list[159]
del index_list[-1]

del name_list[241]
del grade_list[241]
del index_list[-1]

del name_list[251]
del grade_list[251]
del index_list[-1]

del name_list[271]
del grade_list[271]
del index_list[-1]

# %%

image_list = np.zeros(256608000, dtype = np.float32)
image_list = np.reshape(image_list, (240, 240, 4455))

# %%

for i in range(495):
    name = name_list[i]
    file_path = 'D:/Downloads/Aspera/PKG - UCSF-PDGM-v3/UCSF-PDGM-v3/UCSF-PDGM-'+name+'_nifti/UCSF-PDGM-'+name+'_T1.nii.gz'
    file = nib.load(file_path)
    image = file.get_fdata()
    image.astype('float32')
    i_l = []
    maxi = 0
    for j in range(9):
        maxval = np.max(image[:, :, 92+j])
        if maxval > maxi:
            maxi = maxval
        i_l.append(image[:, :, 92+j])
    for img in i_l:
        img = img/maxi
    image = image/np.max(image)
    for k in range(9):
        index = 9*i+k
        image_list[:, :, k] = i_l[k]

# %%

np.save('2d_data', image_list)

b = np.load('2d_data.npy')
