import sys
import fileinput
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix

# change data type
file_array1 = np.loadtxt("ALOI_realistic.txt", dtype=float, delimiter=';')
file_array2 = np.loadtxt("y_ALOI_realistic.txt", dtype=float, delimiter=';')
file_array3 = np.loadtxt("ytype_ALOI_realistic.txt", dtype=float, delimiter=';')
np.savetxt("ALOI_realistic.txt", file_array1, fmt='%.9f', delimiter=';')
np.savetxt("y_ALOI_realistic.txt", file_array2, fmt='%d', delimiter=';')
np.savetxt("ytype_ALOI_realistic.txt", file_array3, fmt='%d', delimiter=';')

# drop duplicates
file_array1 = np.loadtxt("ALOI_realistic.txt", dtype=str, delimiter=';')
file_array2 = np.loadtxt("y_ALOI_realistic.txt", dtype=str, delimiter=';')
file_array3 = np.loadtxt("ytype_ALOI_realistic.txt", dtype=str, delimiter=';')
new_array = [tuple(row) for row in file_array1]
_, unique_indexes = np.unique(new_array, return_index=True, axis=0)
file_array1 = file_array1[unique_indexes]
file_array2 = file_array2[unique_indexes]
file_array3 = file_array3[unique_indexes]
np.savetxt("ALOI_realistic.txt", file_array1, fmt='%s', delimiter=';')
np.savetxt("y_ALOI_realistic.txt", file_array2, fmt='%s', delimiter=';')
np.savetxt("ytype_ALOI_realistic.txt", file_array3, fmt='%s', delimiter=';')

# add ID at the beggining of the file
i = 0
for line in fileinput.FileInput("ALOI_realistic.txt", inplace=1):
    print('%s;%s' % (format(i, '06d'), line), end="")
    i += 1
i = 0
for line in fileinput.FileInput("y_ALOI_realistic.txt", inplace=1):
    print('%s;%s' % (format(i, '06d'), line), end="")
    i += 1
i = 0
for line in fileinput.FileInput("ytype_ALOI_realistic.txt", inplace=1):
    print('%s;%s' % (format(i, '06d'), line), end="")
    i += 1

# shuffle all numpy arrays in the same order
file_array1 = np.loadtxt("ALOI_realistic.txt", dtype=str, delimiter=';')
file_array2 = np.loadtxt("y_ALOI_realistic.txt", dtype=str, delimiter=';')
file_array3 = np.loadtxt("ytype_ALOI_realistic.txt", dtype=str, delimiter=';')
X, y, ytype = shuffle(file_array1, file_array2, file_array3)
np.savetxt("ALOI_realistic.txt", X, fmt="%s", delimiter=';')
np.savetxt("y_ALOI_realistic.txt", y, fmt="%s", delimiter=';')
np.savetxt("ytype_ALOI_realistic.txt", ytype, fmt="%s", delimiter=';')

# X = np.loadtxt("ALOI_realistic.txt", dtype=float, delimiter=';')
# X = X[:, 1:]

# calculate distance matrix for all instances
# distance_matrix_X = distance_matrix(X, X)
# np.savetxt("dm_ALOI_realistic.txt", distance_matrix_X, fmt="%s", delimiter=';')

# calculate linear kernel for OCSVM
# kernel_matrix = np.dot(X, X.T)
# np.savetxt("km_ALOI_realistic.txt", kernel_matrix, fmt="%s", delimiter=';')