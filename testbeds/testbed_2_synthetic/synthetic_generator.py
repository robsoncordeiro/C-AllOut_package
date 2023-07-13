import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.utils import shuffle

# generator hyperparameter
MIN_INLIER_CLUSTER_SIZE = 500
MAX_INLIER_CLUSTER_SIZE = 5000
MED_INLIER_CLUSTER_SIZE = (MIN_INLIER_CLUSTER_SIZE + MAX_INLIER_CLUSTER_SIZE) / 2
LOCAL_OUTLIERS_NUMBER = 20
GLOBAL_OUTLIERS_NUMBER = 20
NUMBER_OF_COLLECTIVE_CLUSTERS = 2
NUMBER_OF_COLLECTIVES_PER_CLUSTERS = 10
NDIM = 10
MIN_INLIER_STD = 10
MAX_INLIER_STD = 50
NUM_INLIER_CLUSTERS = 3

# empirically defined constants
MED_INLIER_STD = (MIN_INLIER_STD + MAX_INLIER_STD) / 2

# large value for global outlier std as almost all points will be near this set global_std
# larger global_std means that we will need to generate less points for getting our global outliers
GLOBAL_STD = MAX_INLIER_STD * 15000

# min and max collective outlier std is based on the min and max inlier std
MIN_COLLECTIVE_STD = MIN_INLIER_STD / (MAX_INLIER_CLUSTER_SIZE * 3 / NUMBER_OF_COLLECTIVES_PER_CLUSTERS)
MAX_COLLECTIVE_STD = MAX_INLIER_STD / (MAX_INLIER_CLUSTER_SIZE * 3 / NUMBER_OF_COLLECTIVES_PER_CLUSTERS)

# distances to filter bad outliers, i.e., outliers too close to each other or to inlier clusters
MIN_DISTANCE_BETWEEN_LOCAL_OUTLIERS = MED_INLIER_STD
MIN_DISTANCE_BETWEEN_GLOBAL_OUTLIERS = MAX_INLIER_STD * 10
MIN_DISTANCE_BETWEEN_GLOBAL_OUTLIERS_AND_INLIERS = MAX_INLIER_STD * 10
MIN_DISTANCE_BETWEEN_INLIER_CLUSTERS = MAX_INLIER_STD * 10

# space boundaries where inlier clusters can fall in
SPACE_RANGE = NUM_INLIER_CLUSTERS * MAX_INLIER_STD * 2

# function for generating an inlier cluster and local/global outliers around it
def generate_cluster(cluster_center, inlier_std, inlier_size, global_std, global_size):
    print("Standard deviation of this cluster: " + str(inlier_std))
    cov_matrix = np.zeros((len(cluster_center), len(cluster_center)), float)
    np.fill_diagonal(cov_matrix, inlier_std)

    coord_distributions = np.random.multivariate_normal(cluster_center, cov_matrix, int(10 * inlier_size))
    inliers = [x for x in coord_distributions if np.linalg.norm(np.array(x) - 
    np.array(cluster_center)) <= inlier_std * 2.5][:inlier_size]
    inliers = np.array(inliers)
    print("Number of inliers in this cluster: " + str(len(inliers)))

    local_cov_matrix = np.zeros((len(cluster_center), len(cluster_center)), float)
    np.fill_diagonal(local_cov_matrix, inlier_std * 20)
    local_coord_distributions = np.random.multivariate_normal(cluster_center, local_cov_matrix,
                                                             int(inlier_std * inlier_size))

    # print("Number of points in local outlier cloud: " + str(len(local_coord_distributions)))
    local_outliers = [x for x in local_coord_distributions if np.linalg.norm(np.array(x) - 
    np.array(cluster_center)) > inlier_std * 2 and np.linalg.norm(np.array(x) - 
    np.array(cluster_center)) <= inlier_std * 3]
    print("Number of potential local outliers for this cluster: " + str(len(local_outliers)))
    local_outliers = local_outliers[:1000]

    # find bad local outliers that are too close to other local outliers
    local_outliers_to_remove = []
    for count, elem in enumerate(local_outliers):
        for count2, elem2 in enumerate(local_outliers):
            if count != count2 and count2 not in local_outliers_to_remove and count not in local_outliers_to_remove:
                # check if current local outlier is close to any other local outlier
                if np.linalg.norm(np.array(elem) - np.array(elem2)) < inlier_std / 3:
                    local_outliers_to_remove.append(count)
                    break

    # remove bad global outliers
    local_outliers = np.array([x for count, x in enumerate(local_outliers) if count not in local_outliers_to_remove])

    print("Number of local outliers in this cluster: " + str(len(local_outliers)))

    global_cov_matrix = np.zeros((len(cluster_center), len(cluster_center)), float)
    np.fill_diagonal(global_cov_matrix, global_std)
    global_coord_distributions = np.random.multivariate_normal(cluster_center, global_cov_matrix, int(10 * global_size)).T
    global_outliers = [x for x in global_coord_distributions.T if np.linalg.norm(np.array(x) - 
    np.array(cluster_center)) > MAX_INLIER_STD]
    global_outliers = np.array(global_outliers)

    return inliers, local_outliers, global_outliers

inliers_stack = []
local_outliers_stack = []
global_outliers_stack = []

# procedure that decides where inlier cluster will be and that calls generate_cluster function 
cluster_center_list = []
for cl in range(NUM_INLIER_CLUSTERS):
    inlier_std = round(random.uniform(MIN_INLIER_STD, MAX_INLIER_STD), 2)
    inlier_size = random.randint(MIN_INLIER_CLUSTER_SIZE, MAX_INLIER_CLUSTER_SIZE)
    cluster_center_coord = []

    current_cluster_center = np.around(np.random.uniform(-SPACE_RANGE, SPACE_RANGE, NDIM), 2)

    ## check if the generated cluster center coordinate is close to any other already defined cluster center coordinate
    count = 0
    while count < len(cluster_center_list):
        diff = np.linalg.norm(np.array(cluster_center_list[count]) - current_cluster_center)

        if diff < MIN_DISTANCE_BETWEEN_INLIER_CLUSTERS:
            count = -1
            current_cluster_center = np.around(np.random.uniform(-SPACE_RANGE, SPACE_RANGE, NDIM), 2)
        count += 1

    cluster_center_list.append(current_cluster_center)
    inliers, local_outliers, global_outliers = generate_cluster(current_cluster_center, inlier_std,
                                                                inlier_size, GLOBAL_STD, GLOBAL_OUTLIERS_NUMBER)
    if len(inliers) > 0:
        inliers_stack.append(inliers)
    if len(local_outliers) > 0:
        local_outliers_stack.append(local_outliers)
    if len(global_outliers) > 0:
        global_outliers_stack.append(global_outliers)
inliers_stack = np.vstack(inliers_stack)

# the local outliers generated for each cluster are stacked and then shuffled so we can pick
# the LOCAL_OUTLIERS_NUMBER amount at random
local_outliers_stack = np.vstack(local_outliers_stack)
local_outliers_stack = shuffle(local_outliers_stack)
local_outliers_stack = local_outliers_stack[:LOCAL_OUTLIERS_NUMBER]
print("Final number of local outliers: " + str(len(local_outliers_stack)))

# the global outliers generated for each cluster are stacked and then shuffled so we can pick
# a subsample at random to filter
global_outliers_stack = np.vstack(global_outliers_stack)
global_outliers_stack = shuffle(global_outliers_stack)
global_outliers_stack = global_outliers_stack[:1000]
print("Potential global outliers: " + str(len(global_outliers_stack)))

# due to their larger std, global outliers may reach and fall inside neighboring inlier clusters.
# Thus, we need to find bad global outliers that are either close to other global outliers or to cluster centers
global_outliers_to_remove = []
for count, elem in enumerate(global_outliers_stack):
    for count2, elem2 in enumerate(global_outliers_stack):
        if count != count2 and count2 not in global_outliers_to_remove and count not in global_outliers_to_remove:
            # check if current global outlier is close to any other global outlier
            if np.linalg.norm(np.array(elem) - np.array(elem2)) < MIN_DISTANCE_BETWEEN_GLOBAL_OUTLIERS:
                global_outliers_to_remove.append(count)
                break
            # check if current global outlier is close to any cluster center
            for cluster_counter in np.arange(0, len(cluster_center_list), 1):
                if np.linalg.norm(np.array(elem) - np.array(cluster_center_list[cluster_counter])) < MIN_DISTANCE_BETWEEN_GLOBAL_OUTLIERS_AND_INLIERS:
                    global_outliers_to_remove.append(count)
                    break


# remove bad global outliers
global_outliers = np.array([x for count, x in enumerate(global_outliers_stack) if count not in global_outliers_to_remove])

# select some global outliers to be collective outliers centers from the already shuffled list of potential global outliers
collective_outlier_centers = [x for x in global_outliers[-NUMBER_OF_COLLECTIVE_CLUSTERS:, :]]

global_outliers = global_outliers[:GLOBAL_OUTLIERS_NUMBER]
print("Final number of global outliers: " + str(len(global_outliers)))

# generate collective outliers around the selected centers
collective_outliers_stack = []
for count, center in enumerate(collective_outlier_centers):
    collective_cov_matrix = np.zeros((len(center), len(center)), float)
    adaptive_collective_std = round(random.uniform(MIN_COLLECTIVE_STD, MAX_COLLECTIVE_STD), 2)
    np.fill_diagonal(collective_cov_matrix, adaptive_collective_std)

    collective_outliers = np.random.multivariate_normal(center, collective_cov_matrix, NUMBER_OF_COLLECTIVES_PER_CLUSTERS)
    collective_outliers_stack.append(collective_outliers)
collective_outliers_stack = np.vstack(collective_outliers_stack)


# these are the plots for each dataset generated

# plt.scatter(inliers_stack[:, 0], inliers_stack[:, 1], s=0.5, c='b', linewidths=0.5, alpha=1)
# plt.scatter(local_outliers_stack[:, 0], local_outliers_stack[:, 1], s=0.5, c='lawngreen', linewidths=0.5, alpha=1)
# plt.scatter(global_outliers[:, 0], global_outliers[:, 1], s=0.5, c='r', linewidths=0.5, alpha=1)
# plt.scatter(collective_outliers_stack[:, 0], collective_outliers_stack[:, 1], s=0.5, c='k', linewidths=0.5, alpha=1)
# plt.axis('equal')
# plt.savefig('S1_A.png')
# plt.show()
# plt.close('all')

# plt.scatter(inliers_stack[:, 0], inliers_stack[:, 1], s=0.5, linewidths=0.5, c='b')
# plt.scatter(local_outliers_stack[:, 0], local_outliers_stack[:, 1], s=0.5, c='lawngreen')
# plt.axis('equal')
# plt.savefig('S1_L.png')
# plt.show()
# plt.close('all')

# plt.scatter(inliers_stack[:, 0], inliers_stack[:, 1], s=0.5, linewidths=0.5, c='b')
# plt.scatter(global_outliers[:, 0], global_outliers[:, 1], s=0.5, c='r')
# plt.axis('equal')
# plt.savefig('S1_G.png')
# plt.show()
# plt.close('all')

# plt.scatter(inliers_stack[:, 0], inliers_stack[:, 1], s=0.5, linewidths=0.5, c='b')
# plt.scatter(collective_outliers_stack[:, 0], collective_outliers_stack[:, 1], s=0.5, c='k')
# plt.axis('equal')
# plt.savefig('S1_C.png')
# plt.show()
# plt.close('all')


# following steps are all related to creating the labels and files with the generated outliers
inlier_labels = [0] * len(inliers_stack) 
local_labels = [1] * len(local_outliers_stack)
global_labels = [2] * len(global_outliers)
collective_labels = [3] * len(collective_outliers_stack)

outlier_labels = [1] * (len(local_outliers_stack) + len(global_outliers) + len(collective_outliers_stack))
labels = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(outlier_labels).reshape(-1, 1)))
labels_by_type = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(local_labels).reshape(-1, 1), np.array(global_labels).reshape(-1, 1), np.array(collective_labels).reshape(-1, 1)))
all_data_stack = np.vstack((inliers_stack, local_outliers_stack, global_outliers, collective_outliers_stack))

np.savetxt("S1/S1_A.txt", all_data_stack, fmt='%.9f', delimiter=';')
np.savetxt("S1/y_S1_A.txt", labels, fmt='%.9f', delimiter=';')
np.savetxt("S1/ytype_S1_A.txt", labels_by_type, fmt='%.9f', delimiter=';')

outlier_labels = [1] * len(local_outliers_stack)
labels = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(outlier_labels).reshape(-1, 1)))
labels_by_type = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(local_labels).reshape(-1, 1)))
all_data_stack = np.vstack((inliers_stack, local_outliers_stack))
np.savetxt("S1/S1_L.txt", all_data_stack, fmt='%.9f', delimiter=';')
np.savetxt("S1/y_S1_L.txt", labels, fmt='%.9f', delimiter=';')
np.savetxt("S1/ytype_S1_L.txt", labels_by_type, fmt='%.9f', delimiter=';')

outlier_labels = [1] * len(global_outliers)
labels = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(outlier_labels).reshape(-1, 1)))
labels_by_type = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(global_labels).reshape(-1, 1)))
all_data_stack = np.vstack((inliers_stack, global_outliers))
np.savetxt("S1/S1_G.txt", all_data_stack, fmt='%.9f', delimiter=';')
np.savetxt("S1/y_S1_G.txt", labels, fmt='%.9f', delimiter=';')
np.savetxt("S1/ytype_S1_G.txt", labels_by_type, fmt='%.9f', delimiter=';')

outlier_labels = [1] * len(collective_outliers_stack)
labels = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(outlier_labels).reshape(-1, 1)))
labels_by_type = np.vstack((np.array(inlier_labels).reshape(-1, 1), np.array(collective_labels).reshape(-1, 1)))
all_data_stack = np.vstack((inliers_stack, collective_outliers_stack))
np.savetxt("S1/S1_C.txt", all_data_stack, fmt='%.9f', delimiter=';')
np.savetxt("S1/y_S1_C.txt", labels, fmt='%.9f', delimiter=';')
np.savetxt("S1/ytype_S1_C.txt", labels_by_type, fmt='%.9f', delimiter=';')