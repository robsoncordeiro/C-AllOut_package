import sys
import fileinput
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix
from scipy.io import loadmat

# convert .mat files to .csv
# mat_file = loadmat("shuttle.mat")
# file_array1 = mat_file['X']
# file_array2 = mat_file['y'].reshape(-1, 1)
# np.savetxt("shuttle2_X.csv", file_array1, delimiter=',')
# np.savetxt("shuttle2_y.csv", file_array2, delimiter=',')

# change data type
file_array1 = np.loadtxt("ALOI_X.csv", dtype=float, delimiter=',')
file_array2 = np.loadtxt("ALOI_y.csv", dtype=float, delimiter=',')
np.savetxt("ALOI.txt", file_array1, fmt='%.9f', delimiter=';')
np.savetxt("y_ALOI.txt", file_array2, fmt='%d', delimiter=';')

# drop duplicates
file_array1 = np.loadtxt("ALOI.txt", dtype=str, delimiter=';')
file_array2 = np.loadtxt("y_ALOI.txt", dtype=str, delimiter=';')
new_array = [tuple(row) for row in file_array1]
_, unique_indexes = np.unique(new_array, return_index=True, axis=0)
file_array1 = file_array1[unique_indexes]
file_array2 = file_array2[unique_indexes]
np.savetxt("ALOI.txt", file_array1, fmt='%s', delimiter=';')
np.savetxt("y_ALOI.txt", file_array2, fmt='%s', delimiter=';')

# scale data to 0 mean and 1 sdv
file_array1 = np.loadtxt("ALOI.txt", dtype=float, delimiter=';')
file_array2 = np.loadtxt("y_ALOI.txt", dtype=float, delimiter=';')
scaler = StandardScaler()
file_array1 = scaler.fit_transform(file_array1)
np.savetxt("ALOI.txt", file_array1, fmt='%.9f', delimiter=';')
np.savetxt("y_ALOI.txt", file_array2, fmt='%d', delimiter=';')

# add ID at the beggining of the file
i = 0
for line in fileinput.FileInput("ALOI.txt", inplace=1):
    print('%s;%s' % (format(i, '06d'), line), end="")
    i += 1
i = 0
for line in fileinput.FileInput("y_ALOI.txt", inplace=1):
    print('%s;%s' % (format(i, '06d'), line), end="")
    i += 1

# shuffle both numpy arrays in the same order
file_array1 = np.loadtxt("ALOI.txt", dtype=str, delimiter=';')
file_array2 = np.loadtxt("y_ALOI.txt", dtype=str, delimiter=';')
X, y = shuffle(file_array1, file_array2)
np.savetxt("ALOI.txt", X, fmt="%s", delimiter=';')
np.savetxt("y_ALOI.txt", y, fmt="%s", delimiter=';')

# X = np.loadtxt("ALOI.txt", dtype=float, delimiter=';')
# X = X[:, 1:]

# calculate distance matrix for all instances
# distance_matrix_X = distance_matrix(X, X)
# np.savetxt("dm_ALOI.txt", distance_matrix_X, fmt="%s", delimiter=';')

# calculate linear kernel for OCSVM
# kernel_matrix = np.dot(X, X.T)
# np.savetxt("km_ALOI.txt", kernel_matrix, fmt="%s", delimiter=';')