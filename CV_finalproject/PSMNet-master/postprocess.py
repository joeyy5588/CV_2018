from util import readPFM
import numpy as np
import cv2
import time
import numpy.matlib
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.sparse import bsr_matrix
from scipy.signal import medfilt2d
from scipy.sparse import coo_matrix
import math


def cal_avgerr(GT, disp):
    return np.sum(np.multiply(np.abs(GT - disp), GT)) / np.sum(GT)
def filtermask(colimg, x, y, winsize,gamma_c,gamma_p):
    radius = math.floor(winsize / 2)
    h, w, c = colimg.shape
    patch_h = min(h - 1, y + radius) - max(0, y - radius) + 1
    patch_w = min(w - 1, x + radius) - max(0, x - radius) + 1
    centercol = colimg[y, x, :].reshape(1, 1, 3)
    centercol1 = centercol.copy()
    for i in range(patch_h - 1):
        centercol = np.vstack((centercol, centercol1))
    centercol1 = centercol.copy()
    for j in range(patch_w - 1):
        centercol = np.hstack((centercol, centercol1))
    #centercol = np.matlib.repmat(centercol, patch_h, patch_w)
    #print(centercol.shape)
    patchYinds = np.arange(max(0, y - radius),  min(h - 1, y + radius) + 1)
    patchYinds = np.matlib.repmat(patchYinds.reshape(-1, 1), 1, patch_w)
    patchXinds = np.arange(max(0, x - radius),  min(w - 1, x + radius) + 1)
    patchXinds = np.matlib.repmat(patchXinds, patch_h, 1)
    curPatch = colimg[max(0, y-radius) : min(h - 1, y+radius) + 1, max(0, x-radius) : min(w - 1, x+radius) + 1, :]
    coldiff = np.sqrt(np.sum((centercol - curPatch) ** 2, axis = 2))
    sdiff = np.sqrt( (x-patchXinds) ** 2 + (y-patchYinds) ** 2 )
    weights = np.exp(-1 * coldiff / (gamma_c*gamma_c))*np.exp(-1 * sdiff / (gamma_p*gamma_p))
    #print(coldiff, sdiff, weights)
    return weights
def weightedMedianMatlab(left_img,disp_img,winsize,gamma_c,gamma_p):
    h, w, c = left_img.shape
    smoothed_left_img1 = medfilt2d(left_img[:, :, 0])
    smoothed_left_img2 = medfilt2d(left_img[:, :, 1])
    smoothed_left_img3 = medfilt2d(left_img[:, :, 2])
    smoothed_left_img = np.dstack((smoothed_left_img1, smoothed_left_img2, smoothed_left_img3))
    radius = math.floor(winsize/2)
    medianFiltered = np.zeros((h,w))
    for y in range(h):
        for x in range(w):
            maskVals = filtermask(smoothed_left_img, x, y, winsize, gamma_c, gamma_p).astype(np.double).flatten()
            dispVals = disp_img[max(0, y-radius) : min(h - 1, y+radius) + 1, max(0, x-radius) : min(w - 1, x+radius) + 1]
            #print(maskVals, dispVals)
            maxDispVal = int(np.max(dispVals))
            col_idx = (dispVals.copy() - 1).astype(np.int).flatten()
            row_idx = np.zeros(dispVals.shape).astype(np.int).flatten()
            #print(maskVals.dtype, row_idx.dtype, col_idx.dtype)
            hist = bsr_matrix((maskVals, (row_idx, col_idx)), shape=(1, maxDispVal)).toarray()
            hist_sum = np.sum(hist)
            hist_cumsum = np.cumsum(hist)
            possbileDispVals = np.arange(1, maxDispVal + 1)
            medianval = possbileDispVals[hist_cumsum > (hist_sum / 2)]
            medianFiltered[y, x] = medianval[0]
    return medianFiltered
def fillPixelsReference(Il, final_labels, gamma_c, gamma_d, r_median, numDisp):
    m,n = final_labels.shape
    occPix = np.zeros((m,n))
    occPix[final_labels < 0] = 1
    fillVals = np.ones((m)) * numDisp
    final_labels_filled = final_labels.copy()
    for col in range(n):
        curCol = final_labels[:,col].copy()
        curCol[curCol == -1] = fillVals[curCol == -1];
        fillVals[curCol != -1] = curCol[curCol != -1];
        final_labels_filled[:, col] = curCol
    fillVals = np.ones((m)) * numDisp
    final_labels_filled1 = final_labels.copy()
    for col in range(n - 1, -1, -1):
        curCol = final_labels[:,col].copy()
        curCol[curCol == -1] = fillVals[curCol == -1];
        fillVals[curCol != -1] = curCol[curCol != -1];
        final_labels_filled1[:, col] = curCol
    final_labels = np.minimum(final_labels_filled,final_labels_filled1)
    #plt.imshow(final_labels, cmap = 'gray')
    #plt.show()
    final_labels_smoothed = weightedMedianMatlab(Il,final_labels,r_median,gamma_c,gamma_d)
    final_labels[occPix == 1] = final_labels_smoothed[occPix == 1]
    #plt.imshow(final_labels, cmap = 'gray')
    #plt.show()
    return final_labels

def main():
	for i in range(10):
		pred_disp = readPFM('output/TL' + str(i) + '.pfm')
		gt = readPFM('data/Synthetic/TLD' + str(i) + '.pfm')
		left_img = cv2.imread('data/Synthetic/TL' + str(i) + '.png')
		loss = cal_avgerr(gt, pred_disp)
		print(loss)
		new_disp = weightedMedianMatlab(left_img, pred_disp, 19, 0.1, 9)
		loss = cal_avgerr(gt, new_disp)
		print(loss)
		disp_normalized = (new_disp * 255.0).astype(np.uint8)
		disp_normalized = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
		cv2.imshow("visualized disparity", disp_normalized)
		cv2.waitKey(10000)
		cv2.destroyAllWindows()

if __name__ == '__main__':
   main()
