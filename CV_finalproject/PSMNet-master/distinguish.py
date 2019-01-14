import cv2
import numpy as np


for i in range(10):
	img = cv2.imread('data/Synthetic/TL' + str(i) + '.png')
	img1 = cv2.imread('data/Real/TL' + str(i) + '.bmp')
	print(img.shape)
	print(img1.shape)