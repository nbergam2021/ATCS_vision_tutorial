
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math


def gaussian(x, mu, sigma):
    coeff = 1/(sigma* (2*math.pi)**(0.5) )
    exponent = -0.5 * math.pow(x-mu,2) / (sigma)**2
    return coeff * math.e**(exponent)

g_filter = [[0.077847, 0.123317,0.077847],[0.123317,0.195346,0.123317],[0.077847, 0.123317,0.077847]]


def i_gauss(a, b):
    interval = 0.01
    s = 0
    marker = a
    while marker <= b:
        s += gaussian(marker,0,1)*interval
        marker += interval
    return sm

def normalize(vector):
    s = 0.0
    for i in range(len(vector)):
        s += vector[i]
    for i in range(len(vector)):
        vector[i] /= s
    return vector

f = normalize([i_gauss(-1.5,-0.5), i_gauss(-0.5,0.5), i_gauss(0.5,1.5)])
F = np.outer(f,f)

            
def convolution(img, flt):
    x = len(img)
    y = len(img[0])
    for i in range(x):
        for j in range(y):
            s = 0
            for a in range(3):
                for b in range(3):
                    if i + a-1 < 0 or i + a-1 >= x or j + b-1 < 0 or j + b-1 >= y:
                        continue
                    else:
                        s += flt[a,b]*img[i + a-1, j + b-1]
            img[i,j] = s
    return img


img = cv.imread('lena512.jpg')
img_final = convolution(img, F)
other = cv.GaussianBlur(img, (3,3), 1.0)

print(img_final - other)

#img_final = cv.cvtColor(img_final, cv.COLOR_BGR2GRAY)
#cv.imwrite('lenaBLURRED.jpg', img_final)
plt.imshow(img_final)
plt.show()



    
