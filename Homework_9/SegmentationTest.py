from ImageSegmentation import segmentation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

Img = mpimg.imread("sample2.png")

#K-Means segmentation
segmentation(Img, "K-MEANS")

#Spectral segmentation
segmentation(Img, "SPECTRAL")

#Mean shift segmentation
segmentation(Img, "MEAN-SHIFT")
