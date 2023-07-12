# utilities for the paper experiments. AUCPR, AUCROC and plotting functions

import numpy as np

def getStats(pred_y, test_y_outlier, test_y_not_outlier):
    overlap_TP = set(pred_y) & set(test_y_outlier)
    TP = len(overlap_TP)

    overlap_TN = set(test_y_not_outlier) - set(pred_y)
    TN = len(overlap_TN)

    overlap_FP = set(pred_y) & set(test_y_not_outlier)
    FP = len(overlap_FP)

    overlap_FN = set(test_y_outlier) - set(pred_y)
    FN = len(overlap_FN)

    return TP, TN, FP, FN

def getPrecisionRecall(TP, TN, FP, FN):
    if TP == 0 and FP == 0 and FN == 0:
        return (1, 1)
    if TP == 0 and FP == 0:
        return (1, 0)
    if TP == 0:
        return (0, 0)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    return (float(format(precision, '.4f')), float(format(recall, '.4f')))

def getROC(TP, TN, FP, FN):
    if TP == 0:
        tpr = 0
    else:
        tpr = TP / (TP + FN)

    if FP == 0:
        fpr = 0
    else:
        fpr = FP / (FP + TN)
    if TP == 0:
        return (0, fpr)
    if FP == 0:
        return (tpr, 0)

    return (float(format(tpr, '.4f')), float(format(fpr, '.4f')))

def saveScores(input_ranking, max_threshold, outlier_labels, inlier_labels, pxr_list_to_append, roc_list_to_append):
    precision_list = [1.0]
    recall_list = [0.0]
    tpr_list = [0.0]
    fpr_list = [0.0]

    for j in np.arange(10, max_threshold + 10, 10):
        outliest = input_ranking[:j, 0]
        outliest = list(map(int, outliest))

        TP, TN, FP, FN = getStats(outliest, outlier_labels, inlier_labels)

        precision, recall = getPrecisionRecall(TP, TN, FP, FN)
        precision_list.append(precision)
        recall_list.append(recall)

        tpr, fpr = getROC(TP, TN, FP, FN)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    pxr_list_to_append.append(float(format(np.trapz(precision_list, recall_list), '.3f')))
    roc_list_to_append.append(float(format(np.trapz(tpr_list, fpr_list), '.3f')))

def getInfoForRankingGraphs(input_ranking, inlier_labels, local_labels, global_labels, collective_labels):
    input_ranking_colors = []
    input_ranking_sizes = []
    for x in input_ranking[:, 0]:
        if x in inlier_labels:
            input_ranking_colors.append('b')
            input_ranking_sizes.append(0.2)
        if x in local_labels:
            input_ranking_colors.append('lawngreen')
            input_ranking_sizes.append(6)
        if x in global_labels:
            input_ranking_colors.append('r')
            input_ranking_sizes.append(6)
        if x in collective_labels:
            input_ranking_colors.append('black')
            input_ranking_sizes.append(6)
    return input_ranking_colors, input_ranking_sizes

def getInfoForRankingGraphsKneeCut(input_ranking, knee):
    input_ranking_colors = []
    input_ranking_sizes = []
    for x in np.arange(input_ranking.shape[0]):
        if x < knee:
            input_ranking_colors.append('y')
            input_ranking_sizes.append(3)
        else:
            input_ranking_colors.append('black')
            input_ranking_sizes.append(3)
    return input_ranking_colors, input_ranking_sizes