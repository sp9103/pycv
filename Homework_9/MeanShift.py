import numpy
import copy

def MeanShift(data, rad):
    label = numpy.zeros(shape=(data.shape[1]))
    CovPos = numpy.zeros(shape=(data.shape))
    Idx = []
    prevIdx = []

    rad = rad*rad

    #Mean-shift algorithm
    for n in range(data.shape[1]):
        #Initialize convergence point
        CovPos[:,n] = copy.copy(data[:,n])
        del Idx[:]
        del prevIdx[:]
        
        while 1:
            prevIdx = copy.copy(Idx[:])
            del Idx[:]
            
            for i in range(data.shape[1]):
                #euclidean distance calculation
                sub = CovPos[:,n] - data[:,i]
                dist = numpy.dot(sub, sub)
                
                if dist <= rad:
                    Idx.append(i)

            #convergence check
            if prevIdx == Idx:
                break
            if len(Idx) == 0:
                break

            #Center search
            CovPos[:,n] = 0.0
            for j in range(len(Idx)):
                CovPos[:,n] += data[:, Idx[j]]
            CovPos[:,n] /= float(len(Idx))
                
    #Merge Converge Point
    cCount = 1
    for n in range(len(label)):
        if label[n] == 0:
            label[n] = cCount
            for i in range(len(label)):
                if label[i] != 0:
                    continue
                #euclidean distance calculation
                sub = CovPos[:,n] - CovPos[:,i]
                dist = numpy.sqrt(numpy.dot(sub, sub))
                if dist <= 0.01:
                    label[i] = cCount
            cCount+=1
            
    return label
