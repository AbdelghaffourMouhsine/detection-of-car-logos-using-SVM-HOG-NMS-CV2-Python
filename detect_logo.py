#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

def calcule_Descripteurs_logo(img):
    resized_img_test_1 = cv2.resize(img, (200, 200))
    gray_test_1 = cv2.cvtColor(resized_img_test_1, cv2.COLOR_BGR2GRAY)
    fd_test_1, hog_image = hog(gray_test_1, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
    fd_test_1 = fd_test_1.reshape(1,-1)
    return fd_test_1

def calcule_Descripteurs_is_logo(img_test_1):
    #img_filtre = cv2.GaussianBlur(img_test_1, (3,3), 0) # noise reduction
    resized_img_test_1 = cv2.resize(img_test_1, (128, 128))
    gray_test_1 = cv2.cvtColor(resized_img_test_1, cv2.COLOR_BGR2GRAY)
    #gray_test_1 = cv2.equalizeHist(gray_test_1) 
    fd_test_1, hog_image = hog(gray_test_1, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, channel_axis=None)
    fd_test_1 = fd_test_1.reshape(1,-1)
    return fd_test_1

# %%
