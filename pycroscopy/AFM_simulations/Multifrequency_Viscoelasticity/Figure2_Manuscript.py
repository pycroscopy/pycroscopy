# -*- coding: utf-8 -*-
"""
Created on Sat Dec 02 21:05:09 2017

@author: Enrique Alejandro
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from AFM_calculations import figannotate

path = os.getcwd()

os.chdir(path+'/1stMode')
first = pd.read_csv('1stMode.txt', delimiter = '\t')
amp_ratio1 = first.iloc[:,0].values
peakf1 = first.iloc[:,1].values
depth1 = first.iloc[:,2].values

#plt.plot(amp_ratio1, peakf1)
os.chdir(path+'/Bimodal')
bimodal = pd.read_csv('Bimodal.txt', delimiter ='\t')
amp_ratiobi = bimodal.iloc[:,0].values
peakf_bi = bimodal.iloc[:,1].values
depth_bi = bimodal.iloc[:,2].values

os.chdir(path + '/2ndMode')
second = pd.read_csv('2ndMode.txt', delimiter ='\t')
amp_ratio2 = second.iloc[:,0].values
peakf2= second.iloc[:,1].values
depth2 = second.iloc[:,2].values

fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,3))
ax0.plot(amp_ratio1, peakf1, 'r', lw=3, label='1st')
ax0.plot(amp_ratio2, peakf2, 'b', lw=3, label='2nd')
ax0.plot(amp_ratiobi, peakf_bi, 'g', lw=3, label='1st+2nd')
#ax0.legend(loc=2)
ax0.set_xlabel(r'Amplitude ratio', color='k', fontsize='18',fontweight='bold')
ax0.set_ylabel('Peak Force (nN)', color='k', fontsize='18',fontweight='bold')


ax1.plot(amp_ratio1, depth1,'r', lw=3)
ax1.plot(amp_ratio2, depth2, 'b', lw=3)
ax1.plot(amp_ratiobi, depth_bi, 'g', lw=3)
ax1.set_xlabel(r'Amplitude ratio', color='k', fontsize='18',fontweight='bold')
ax1.set_ylabel('Penetration \n Depth (nm)', color='k', fontsize='18',fontweight='bold')

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,  wspace=0.3, hspace=None)

figannotate(text='(a)', fs=20, pos=(0.05,1.2))
os.chdir(path)
#plt.savefig('Fig2_simuls.png', bbox_inches='tight', dpi=900)
