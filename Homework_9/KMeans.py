import numpy
import random
import copy

def K_Means(data, nClass):
    label = numpy.zeros(shape=(data.shape[1]))
    prvLabel = numpy.zeros(shape=(data.shape[1]))
    nLabel = numpy.zeros(shape=(nClass))
    CovPos = numpy.zeros(shape=(data.shape[0], nClass))
    tCov = numpy.zeros(shape=(data.shape[0], nClass))

    
    #init random point
    tsize = tCov.shape
    for row in range(tsize[0]):
        for col in range(tsize[1]):
            tCov[row][col] = random.random()

    #Loop until convergence
    while 1:
        CovPos = copy.copy(tCov)
        tCov[:] = 0.0
        nLabel[:] = 0.0

        for i in range(data.shape[1]):
            Mindist = 9999.0
            for j in range(nClass):
                sub = CovPos[:,j] - data[:,i]
                dist = numpy.dot(sub, sub)

                if Mindist > dist:
                    Mindist = dist
                    label[i] = j

            tCov[:,label[i]] += data[:,i]
            nLabel[label[i]] += 1

        if label.all() == prvLabel.all():
            break

        for i in range(tCov.shape[1]):
            if nLabel[i] != 0:
                tCov[:,i] /= nLabel[i]
        prvLabel = label[:] 

    label[:] += 1

    return label
