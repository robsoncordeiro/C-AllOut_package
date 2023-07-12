import datasets_list
from experiments_utils import getStats
from experiments_utils import getPrecisionRecall
from experiments_utils import getROC
import numpy as np
from sklearn.neighbors import NearestNeighbors
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from isolation_simforest import IsolationSimilarityForest

# competitors hyperparameters for Testbed 1
LOF_k = [1, 3, 5, 10, 15, 30, 50, 75, 100]
kNN_k = [1, 3, 5, 10, 15, 30, 50, 75, 100]
OCSVM_nu = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

for [benchmark_path, benchmark_name, y_benchmark_name, benchmark_data_lines, benchmark_data_columns, benchmark_object_byte_size] in datasets_list.datasets_benchmark:
    # separate the labels in two lists, one with all the outliers (y = 1) and one with all the inliers (y = 0)
    f=open(y_benchmark_name, "r")
    lines=f.readlines()
    test_y_outlier=[]
    test_y_not_outlier=[]
    for x in lines:
        x_spl = x.split(';')[1]
        if x_spl.rstrip() == '0':
            test_y_not_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '1':
            test_y_outlier.append(int(x.split(';')[0]))
    f.close()

    # lists for saving AUCPR and AUCROC results
    lof_aucpr_list = []
    lof_aucroc_list = []
    knn_aucpr_list = []
    knn_aucroc_list = []
    ocsvm_aucpr_list = []
    ocsvm_aucroc_list = []
    sif_aucpr_list = []
    sif_aucroc_list = []

    # precalculated distance matrix used to speed up some competitors running time
    distance_matrix = np.loadtxt("dm_" + benchmark_name + ".txt", dtype=float, delimiter=';')

    # instance IDs used for performance evaluation
    ids = np.loadtxt(y_benchmark_name, dtype=int, delimiter=';')
    ids = ids[:, 0]

    # LOF evaluation
    for k in LOF_k:
        model = LOF(n_neighbors=k, metric='precomputed')
        model.fit(distance_matrix)
        scores = model.decision_scores_
        scores = np.column_stack((ids, scores))
        scores = scores[scores[:, 1].argsort()[::-1]]

        precision_list = [1.0]
        recall_list = [0.0]
        tpr_list = [0.0]
        fpr_list = [0.0]
        for j in np.arange(1, scores.shape[0] + 1, 1):
            outliest = scores[:j, 0]
            outliest = list(map(int, outliest))

            TP, TN, FP, FN = getStats(outliest, test_y_outlier, test_y_not_outlier)

            precision, recall = getPrecisionRecall(TP, TN, FP, FN)
            precision_list.append(precision)
            recall_list.append(recall)

            tpr, fpr = getROC(TP, TN, FP, FN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        lof_aucpr_list.append(float(format(np.trapz(precision_list, recall_list), '.3f')))
        lof_aucroc_list.append(float(format(np.trapz(tpr_list, fpr_list), '.3f')))

    # kNN evaluation
    for k in kNN_k:
        model = NearestNeighbors(n_neighbors=k, metric='precomputed')
        model.fit(distance_matrix)
        scores, _ = model.kneighbors()
        scores = scores[:, -1]
        scores = scores.ravel()
        scores = np.column_stack((ids, scores))
        scores = scores[scores[:, 1].argsort()[::-1]]

        precision_list = [1.0]
        recall_list = [0.0]
        tpr_list = [0.0]
        fpr_list = [0.0]
        for j in np.arange(1, scores.shape[0] + 1, 1):
            outliest = scores[:j, 0]
            outliest = list(map(int, outliest))

            TP, TN, FP, FN = getStats(outliest, test_y_outlier, test_y_not_outlier)

            precision, recall = getPrecisionRecall(TP, TN, FP, FN)
            precision_list.append(precision)
            recall_list.append(recall)

            tpr, fpr = getROC(TP, TN, FP, FN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        knn_aucpr_list.append(float(format(np.trapz(precision_list, recall_list), '.3f')))
        knn_aucroc_list.append(float(format(np.trapz(tpr_list, fpr_list), '.3f')))

    # precalculated kernel matrix as needed by OCSVM. This is the linear kernel matrix, i.e., the
    # dot product between the instances
    kernel_matrix = np.loadtxt("km_" + benchmark_name + ".txt", dtype=float, delimiter=';')

    # OCSVM evaluation
    for nu in OCSVM_nu:
        model = OCSVM(nu=nu, kernel='precomputed')
        model.fit(kernel_matrix)
        scores = model.decision_scores_
        scores = np.column_stack((ids, scores))
        scores = scores[scores[:, 1].argsort()[::-1]]

        precision_list = [1.0]
        recall_list = [0.0]
        tpr_list = [0.0]
        fpr_list = [0.0]
        for j in np.arange(1, scores.shape[0] + 1, 1):
            outliest = scores[:j, 0]
            outliest = list(map(int, outliest))

            TP, TN, FP, FN = getStats(outliest, test_y_outlier, test_y_not_outlier)

            precision, recall = getPrecisionRecall(TP, TN, FP, FN)
            precision_list.append(precision)
            recall_list.append(recall)

            tpr, fpr = getROC(TP, TN, FP, FN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        ocsvm_aucpr_list.append(float(format(np.trapz(precision_list, recall_list), '.3f')))
        ocsvm_aucroc_list.append(float(format(np.trapz(tpr_list, fpr_list), '.3f')))

    # the implementation of SIF used for the experiments do not support distance matrix as input,
    # so we give the actual instances
    features = np.loadtxt(benchmark_name + ".txt", dtype=float, delimiter=';')
    features = features[:, 1:]

    # SIF is not deterministic, so we need these lists to take the average of 10 executions
    avg_pxr_sif = []
    avg_roc_sif = []
    avg_time_sif = []

    # SIF evaluation
    for sif_iter in np.arange(10):
        model = IsolationSimilarityForest(n_estimators=100)
        model.fit(features)
        scores = model.score_samples(features)
        scores = np.column_stack((ids, scores))
        scores = scores[scores[:, 1].argsort()]

        precision_list = [1.0]
        recall_list = [0.0]
        tpr_list = [0.0]
        fpr_list = [0.0]
        for j in np.arange(1, scores.shape[0] + 1, 1):
            outliest = scores[:j, 0]
            outliest = list(map(int, outliest))

            TP, TN, FP, FN = getStats(outliest, test_y_outlier, test_y_not_outlier)

            precision, recall = getPrecisionRecall(TP, TN, FP, FN)
            precision_list.append(precision)
            recall_list.append(recall)

            tpr, fpr = getROC(TP, TN, FP, FN)
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        avg_pxr_sif.append(float(format(np.trapz(precision_list, recall_list), '.3f')))
        avg_roc_sif.append(float(format(np.trapz(tpr_list, fpr_list), '.3f')))
    sif_aucpr_list.append(np.mean(avg_pxr_sif))
    sif_aucroc_list.append(np.mean(avg_roc_sif))

    # saving performance measurements to disk
    np.savetxt(("./stats/" + benchmark_name + "lof_pxr.csv"), lof_aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "lof_roc.csv"), lof_aucroc_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "knn_pxr.csv"), knn_aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "knn_roc.csv"), knn_aucroc_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "ocsvm_pxr.csv"), ocsvm_aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "ocsvm_roc.csv"), ocsvm_aucroc_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "sif_pxr.csv"), sif_aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "sif_roc.csv"), sif_aucroc_list, fmt='%s', delimiter=',')