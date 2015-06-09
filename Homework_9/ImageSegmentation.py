import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from MeanShift import MeanShift
from KMeans import K_Means
from SpectralClustering import Spectral
import numpy

def segmentation(img, method):
    #Image to Nomalized vector
    size = img.shape
    data = numpy.zeros(shape=(size[2]+1, size[0]*size[1]))
    label = numpy.zeros(shape=(data.shape[1]))
    result = numpy.zeros(shape=(img.shape))

    for row in range(size[0]):
        for col in range(size[1]):
            #Normalize pixel position 0.0~1.0
            data[0][row*size[1]+col] = float(row) / float(size[0]-1)
            data[1][row*size[1]+col] = float(col) / float(size[1]-1)
            for color in range(size[2]-1):
                data[color+2][row*size[1]+col] = img[row][col][color]

    #clustring - input : ndarray(colum vector)
    if method == "MEAN-SHIFT":
       label = MeanShift(data, 0.4)     #radius 0.4
    elif method == "K-MEANS":
       label = K_Means(data, 8)         #8 cluster
    elif method == "SPECTRAL":
       label = Spectral(data, 8)       s #8 cluster

    #clustring result to image
    for row in range(size[0]):
       for col in range(size[1]):
           result[row][col][0] = float(label[row*size[1]+col] * 29 % 100) / 100.0
           result[row][col][1] = float(label[row*size[1]+col] * 13 % 100) / 100.0
           result[row][col][2] = float(label[row*size[1]+col] * 37 % 100) / 100.0
           result[row][col][3] = 1.0

    plt.imsave("result.png", result)
    plt.imshow(result)
    plt.show()

    
    return 0
