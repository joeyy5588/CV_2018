import numpy as np
from numpy import linalg as LA
import cv2
import matplotlib.pyplot as plt 
import sklearn.metrics
import sklearn.manifold
from sklearn.neighbors import KNeighborsClassifier
import sys


train_data = np.zeros((2576, 1))
test_data = np.zeros((2576, 1))
label = []
trainlabel = []

for i in range(1, 41):
	for j in range(1, 8):
		temp = cv2.imread(sys.argv[1] + '/' +  str(i) + '_' + str(j) + '.png', 0).reshape((-1,1))
		train_data = np.hstack((train_data, temp))
		trainlabel.append(i)
	for j in range(8, 11):
		temp = cv2.imread(sys.argv[1] + '/' + str(i) + '_' + str(j) + '.png', 0).reshape((-1,1))
		test_data = np.hstack((test_data, temp))
		label.append(i)

trainlabel = np.array(trainlabel)
label = np.array(label)
train_data = np.delete(train_data, 0, 1)
test_data = np.delete(test_data, 0, 1)
mean_face = np.mean(train_data, axis = 1, keepdims = True)

cov_mat = np.cov((train_data - mean_face))
W, V = LA.eigh(cov_mat)
W = np.flip(W, axis = 0)
V = np.flip(V, axis = 1)
V1 = V[:, :(280 - 40)]
P = np.dot(V1.transpose(), train_data - mean_face)
g_mean = np.mean(P, axis = 1, keepdims = True)
Sw, Sb = 0, 0
for i in range(40):
	P1 = P[:, 7 * i: 7 * (i+1)]
	l_mean = np.mean(P1, axis = 1, keepdims = True)
	sb = (l_mean - g_mean) * (l_mean - g_mean).transpose()
	Sb += sb
	for j in range(7):
		sw = (P1[:, j] - l_mean) * (P1[:, j] - l_mean).transpose()
		Sw += sw

LDA_W, LDA_V = LA.eig(np.dot(LA.inv(Sw), Sb))
sort_w = np.argsort(LDA_W)[::-1]
LDA_W = LDA_W[sort_w]
LDA_V = LDA_V[:, sort_w]
LDA_V = np.real(LDA_V)
LDA_V = LDA_V[:, : 39]
F = np.dot(V1, LDA_V)

f = F[:, 1]
f = f.reshape(56, 46)
plt.imsave(sys.argv[2], f, cmap = 'gray')
plt.imshow(f, cmap = 'gray')
plt.show()