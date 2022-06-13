# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:47:54 2019

@author: Carlos Mendoza R.
"""

"MODELO ESTOCASTICO GENERAL"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pandas as pd

N= 30

S=np.array([[103/169,8/169,1/169,0,57/169],
            [46/194,114/194,22/194,10/194,2/194],
            [11/50,18/50,18/50,2/50,1/50],
            [0,6/14,2/14,2/14,4/14],
            [0,0,0,0,1]])
v = np.array([0,0,1,0,0])

#----------------------------------------------#
###################  CMPD #####################
#----------------------------------------------#

def CMPD(N,P,inicial):  #ec. de  komogorov 
    Pi=np.zeros((N+1,len(P)))
    for n in range(N+1):
        if n==0:
            Pi[n]=inicial
        else:
            Pi[n]=np.dot(P,Pi[n-1])
    return Pi

P1 = CMPD(N,S.T,v)  #CMPD del archivo

plt.plot(np.arange(0,31,1),P1.T[0],label = 'A')
plt.plot(np.arange(0,31,1),P1.T[1],label = 'B')
plt.plot(np.arange(0,31,1),P1.T[2],label = 'C')
plt.plot(np.arange(0,31,1),P1.T[3],label = 'D')
plt.plot(np.arange(0,31,1),P1.T[4],label = 'E')
plt.legend()
plt.show()

print(P1[10])
#----------------------------------------------#
########### GRAFICO 3D ####################
#----------------------------------------------#
#fig = plt.figure(figsize=(10,7))#figsize=(35,30))
#ax = fig.add_subplot(111,projection='3d') 
#ax.set_xlabel(' Tiempo N')
#ax.set_ylabel('Estado S')
#ax.set_zlabel('Probabilidad')
#ax.set_title('Proceso estocastico')
#i=0
#for j in P1:
#    xpos=np.zeros(len(j))+i
#    ypos=np.arange(0,len(j))
#    ancho=np.zeros(len(j))+0.2
#    ax.bar3d(xpos,ypos,0,ancho,ancho,j)
#    i=i+1
#plt.show()
