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

'''mean_face = mean_face.reshape(56,46)
plt.imshow(mean_face, cmap='gray')
plt.show()
mean_face = mean_face.reshape(2576, 1)'''

cov_mat = np.cov((train_data - mean_face))
W, V = LA.eigh(cov_mat)
W = np.flip(W, axis = 0)
V = np.flip(V, axis = 1)

test_img = cv2.imread(sys.argv[2], 0).reshape((-1,1))
P = np.dot(V[:, :].transpose(), test_img - mean_face)
X = np.dot(V[:, :], P) + mean_face
X = X.reshape(56, 46)
plt.imsave(sys.argv[3], X, cmap = 'gray')
#plt.imshow(X, cmap = 'gray')
#plt.show()

'''
#plot mean_face
for i in range(5):
	img = V[:, i]
	img = img.reshape(56, 46)
	plt.imshow(img, cmap = 'gray')
	plt.show()
'''
'''
#2(a)_2
test_img = cv2.imread('hw2-2_data/' + str(8) + '_' + str(6) + '.png', 0).reshape((-1,1))
n_list = [5, 50, 150, 2576]
for n in n_list:
	P = np.dot(V[:, :n].transpose(), test_img - mean_face)
	X = np.dot(V[:, :n], P) + mean_face
	MSE = sklearn.metrics.mean_squared_error(test_img, X)
	
	print(MSE)
	X = X.reshape(56, 46)
	plt.imshow(X, cmap = 'gray')
	plt.show()
	'''

#2(a)_3
'''
P = np.dot(V[:, :100].transpose(), test_data - mean_face)
X = np.dot(V[:, :100], P) + mean_face
print(P.shape, X.shape)
em = sklearn.manifold.TSNE(n_components = 2).fit_transform(P.transpose())
vis_x = em[:, 0]
vis_y = em[:, 1]
plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("jet", 40))
plt.colorbar(ticks=range(40))
plt.clim(0, 40)
plt.title('T-SNE Visualization PCA dim = 100')
plt.show()
'''
#2(b)_1
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
LDA_V = LDA_V[:, : 40]
F = np.dot(V1, LDA_V)

'''
for i in range(5):
	f = F[:, i]
	print(f)
	f = f.reshape(56, 46)
	plt.imshow(f, cmap = 'gray')
	plt.imsave('hw2-2_output/fisherface_' + str(i), f, cmap = 'gray')
	plt.show()
'''
'''
#2(b)_2

low_dim_test = np.dot(V[:, :240].transpose(), test_data - mean_face)
P = np.dot(LDA_V[:, :30].transpose(), low_dim_test)
em = sklearn.manifold.TSNE(n_components = 2).fit_transform(P.transpose())
vis_x = em[:, 0]
vis_y = em[:, 1]
plt.scatter(vis_x, vis_y, c=label, cmap=plt.cm.get_cmap("jet", 40))
plt.colorbar(ticks=range(40))
plt.title('T-SNE Visualization LDA dim = 30')
plt.show()
'''

#2(c)
'''
k_list = [1, 3, 5]
n_list = [3, 10, 39]
slicetrain = [0,0,0]
slicelabel = [0,0,0]
randidx = np.random.randint(len(train_data[0]), size = 150)
slicetrain[0] = train_data[:, randidx]
slicelabel[0] = trainlabel[randidx]
randidx = np.random.randint(len(train_data[0]), size = 150)
slicetrain[1] = train_data[:, randidx]
slicelabel[1] = trainlabel[randidx]
randidx = np.random.randint(len(train_data[0]), size = 150)
slicetrain[2] = train_data[:, randidx]
slicelabel[2] = trainlabel[randidx]
slicetrain = np.array(slicetrain)
slicelabel = np.array(slicelabel)

for k in k_list:
	for n in n_list:
		r_list = [[1,2],[0,2],[0,1]]
		for j in range(3):
			v1, v2 = r_list[j]

			low_dim_test = np.dot(V[:, :240].transpose(), slicetrain[j] - mean_face)
			P_pca = np.dot(V[:, :n].transpose(), slicetrain[j] - mean_face)
			P_lda = np.dot(LDA_V[:, :n].transpose(), low_dim_test)
			neigh_pca = KNeighborsClassifier(n_neighbors=k)
			neigh_pca.fit(P_pca.transpose(), slicelabel[j])
			neigh_lda = KNeighborsClassifier(n_neighbors=k)
			neigh_lda.fit(P_lda.transpose(), slicelabel[j])

			v1_low_dim_test = np.dot(V[:, :240].transpose(), slicetrain[v1] - mean_face)
			v1_pca = np.dot(V[:, :n].transpose(), slicetrain[v1] - mean_face).transpose()
			v1_lda = np.dot(LDA_V[:, :n].transpose(), v1_low_dim_test).transpose()
			v2_low_dim_test = np.dot(V[:, :240].transpose(), slicetrain[v2] - mean_face)
			v2_pca = np.dot(V[:, :n].transpose(), slicetrain[v2] - mean_face).transpose()
			v2_lda = np.dot(LDA_V[:, :n].transpose(), v2_low_dim_test).transpose()

			print('v1', j, k, n, 'pca', np.sum(slicelabel[v1] == neigh_pca.predict(v1_pca))*100/150)
			print('v1', j, k, n, 'lda', np.sum(slicelabel[v1] == neigh_lda.predict(v1_lda))*100/150)
			print('v2', j, k, n, 'pca', np.sum(slicelabel[v2] == neigh_pca.predict(v2_pca))*100/150)
			print('v2', j, k, n, 'lda', np.sum(slicelabel[v2] == neigh_lda.predict(v2_lda))*100/150)

			t_low_dim_test = np.dot(V[:, :240].transpose(), test_data - mean_face)
			t_pca = np.dot(V[:, :n].transpose(), test_data - mean_face).transpose()
			t_lda = np.dot(LDA_V[:, :n].transpose(), t_low_dim_test).transpose()
			print('t', j, k, n, 'pca', np.sum(label == neigh_pca.predict(t_pca))*100/120)
			print('t', j, k, n, 'lda', np.sum(label == neigh_lda.predict(t_lda))*100/120)

'''

