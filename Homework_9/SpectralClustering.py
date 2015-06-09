import numpy
import math
from scipy import linalg as LA

def Spectral(data, nClass):
    label = numpy.zeros(shape=(data.shape[1]))
    sigma = 1.0

    #Affinity, D, Dinv matrix calculation
    sigma = sigma * sigma
    W = numpy.zeros(shape=(data.shape[1], data.shape[1]))
    D = numpy.zeros(shape=(W.shape))
    invD = numpy.zeros(shape=(W.shape))
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            dist = data[:,i] - data[:,j]
            dist = numpy.dot(dist,dist)
            W[i][j] = numpy.exp(-dist / (2.0*sigma))

            D[i][i] += W[i][j]
            invD[i][i] = numpy.sqrt(1.0/D[i][i])
        print(i)

    #ndarray to matrix
    W = numpy.matrix(numpy.array(W))
    D = numpy.matrix(numpy.array(D))
    invD = numpy.matrix(numpy.array(invD))

    #Calculate D^(-0.5)*(D-W)*D^(-0.5)
    T = invD*(D-W)*D

    #eigen vector & eigen value
    val, vec = LA.eigh(T)

    #Calculate label
    y = invD*vec
    #for i in range(data.shape[1]):
    #    for j in range(nClass-1):
    #        if y[i, j+1] > 0: 
    #            label[j] += numpy.power(2,j)
            
    for i in range(nClass-1):
        for j in range(data.shape[1]):
            if y[j,i+1] > 0:        #check row vector or col vector
                label[j] += numpy.power(2,i)
        

    return label
