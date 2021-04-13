
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('sample.png')
print(img.item(0,0,2))
plt.imshow(img)
plt.show()
