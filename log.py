import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import math
from skimage.feature import blob_dog, blob_log, blob_doh

# Raw Filters
gauss = np.asarray([[0.077847, 0.123317,0.077847],[0.123317,0.195346,0.123317],[0.077847, 0.123317,0.077847]])
laplacian = np.asarray([[0.0,-1.0,0.0],[-1.0,4.0,-1.0],[0.0,-1.0,0.0]])
laplacian2 = np.asarray([[-1.0,-1.0,-1.0],[-1.0,8.0,-1.0],[-1.0,-1.0,-1.0]])
log =  np.asarray([[0,1,1,2,2,2,1,1,0],[1,2,4,5,5,5,4,2,1],[1,4,5,3,0,3,5,4,1],[2,5,3,-12,-24,-12,3,5,2],[2,5,0,-14,-40,-24,0,5,2],[2,5,3,-12,-24,-12,3,5,2],[1,4,5,3,0,3,5,4,1],[1,2,4,5,5,5,4,2,1],[0,1,1,2,2,2,1,1,0]])

# Helper Functions

def gaussian(x, mu, sigma):
    coeff = 1/(sigma* (2*math.pi)**(0.5) )
    exponent = -0.5 * math.pow(x-mu,2) / (sigma)**2
    return coeff * math.e**(exponent)

def i_gauss(a, b, sigma):
    interval = 0.01
    s = 0
    marker = a
    while marker <= b:
        s += gaussian(marker,0,sigma)*interval
        marker += interval
    return s

def log_func(x,sigma):
    exp_stuff = -(x*x)/(2*(sigma**2))
    coeff = -(1+exp_stuff)/(math.pi * (sigma**4))
    return coeff*math.exp(exp_stuff)

def i_log_func(a,b,sigma):
    interval = 0.01
    s = 0
    marker = a
    while marker <= b:
        s += log_func(marker,sigma)*interval
        marker += interval
    return s

def create_gaussian_filter(size, sigma):
    f = []
    marker = -size/2.0
    for i in range(size):
        f.append(i_gauss(marker+i,marker+i+1,sigma))
    F = np.outer(f,f)
    return np.true_divide(F,np.sum(F))

def create_log_filter(size, sigma):
    f = []
    marker = -size/2.0
    for i in range(size):
        f.append(i_log_func(marker+i, marker+i+1,sigma))
    F = np.outer(f,f)
    return np.true_divide(F,np.sum(F))

def convolution(img, flt, size):
    x = len(img)
    y = len(img[0])
    ret = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            for a in range(size):
                for b in range(size):
                    if i + a-1 < 0 or i + a-1 >= x or j + b-1 < 0 or j + b-1 >= y:
                        continue
                    else:
                        ret[i,j] += flt[a,b]*img[i + a-1, j + b-1]
    return ret

# My own Gaussian Blur function
def GaussBlur(img, sigma, k_size):
    return convolution(img, create_gaussian_filter(k_size, sigma), k_size)


# The Blob Filters

# Blurring first then Laplacian, using OpenCV (fastest)
def laplacian_of_gaussian(img):
    #blurred = cv.GaussianBlur(img,(5,5),6)
    return cv.Laplacian(img, cv.CV_64F, ksize=5)

# Directly applying the Laplacian of Gaussian filter
def laplacian_of_gaussian_alt1(img):
    return convolution(img, create_log_filter(9,1.5), 5)

# Blurring first then Laplacian, without OpenCV
def laplacian_of_gaussian_alt2(img):
    blurred = convolution(img,gauss,3)
    return convolution(blurred, laplacian, 3)

# DoG: Difference of Gaussians to approximate Laplacian
def difference_of_gaussian(img, s1,s2):
    return cv.GaussianBlur(img,(5,5),s1) - cv.GaussianBlur(img,(5,5),s2)

# DoH: wanted to figure this out but did not have time sadly
def determinant_of_hessian(img):
    return img

# Image Processing + Testing

new_gauss1 = create_gaussian_filter(3, 1.5)
print(new_gauss1)
new_gauss2 = create_gaussian_filter(3, 4)

# Test Images
img1 = cv.imread('lena512.jpg')
img2 = cv.imread('butterfly.jpg',cv.IMREAD_GRAYSCALE)
img3 = cv.imread('persistence_of_memory.jpg',cv.IMREAD_GRAYSCALE)

cv.imwrite("test.png", laplacian_of_gaussian_alt1(img2))

cv.imwrite("results/img1_log_5.jpg", laplacian_of_gaussian(img1))
cv.imwrite("results/img2_log_5.jpg", laplacian_of_gaussian(img2))
cv.imwrite("results/img3_log_5.jpg", laplacian_of_gaussian(img3))

"""
cv.imwrite("results/img1_dog_3.jpg", difference_of_gaussian(img1))
cv.imwrite("results/img2_dog_3.jpg", difference_of_gaussian(img2))
cv.imwrite("results/img3_dog_3.jpg", difference_of_gaussian(img3))

cv.imwrite("results/img1_doh.jpg", determinant_of_hessian(img1))
cv.imwrite("results/img2_doh.jpg", determinant_of_hessian(img2))
cv.imwrite("results/img3_doh.jpg", determinant_of_hessian(img3))
"""
