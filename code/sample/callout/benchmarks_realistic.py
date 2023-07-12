import datasets_list
import os
import numpy as np
from experiments_utils import saveScores
from sklearn.utils import shuffle
from kneed import KneeLocator

# the constant header size in bytes for a Slim-tree node. 
# The header of a node only stores metadata about the node
NODE_HEADER_SIZE = 10

for [benchmark_path, benchmark_name, y_benchmark_name, benchmark_data_lines, benchmark_data_columns, benchmark_object_byte_size] in datasets_list.datasets_realistic:
    # Separate the labels in four lists, one with all the inliers (y = 0), one with local outliers (y = 1)
    # one with global outliers (y = 2) and one with collective outliers (y = 3)
    f=open(benchmark_path + y_benchmark_name, "r")
    lines=f.readlines()
    test_y_outlier=[]
    test_y_not_outlier=[]
    test_y_local_outlier=[]
    test_y_global_outlier=[]
    test_y_collective_outlier=[]
    for x in lines:
        x_spl = x.split(';')[1]
        if x_spl.rstrip() == '0':
            test_y_not_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '1':
            test_y_outlier.append(int(x.split(';')[0]))
            test_y_local_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '2':
            test_y_outlier.append(int(x.split(';')[0]))
            test_y_global_outlier.append(int(x.split(';')[0]))
        if x_spl.rstrip() == '3':
            test_y_outlier.append(int(x.split(';')[0]))
            test_y_collective_outlier.append(int(x.split(';')[0]))
    f.close()

    ## initialize list of 9 independent lists
    aucpr_list_alpha = [[] for i in range(9)]
    aucroc_list_alpha = [[] for i in range(9)]
    aucpr_list = []
    aucroc_list = []
    overall_aucpr_list = []
    overall_aucroc_list = []

    # select the recommended value for the node capacity (or page size) of Slim-tree's leaf nodes
    dataset_size = float(benchmark_data_lines)
    recommended_size = 6
    pages = []
    for max_objects in np.arange(6, int(np.around(dataset_size / 10)), 1):
        lower_bound = (dataset_size / max_objects) / max_objects
        upper_bound = (dataset_size / max_objects)

        if lower_bound < max_objects and max_objects < upper_bound:
            recommended_size = max_objects
            pages.append(2 * recommended_size)
    
    # loop iterating through as many page sizes as needed. In this case, only the recommended page size is used
    for page in [pages[0]]:
        pg = int(benchmark_object_byte_size) * page + NODE_HEADER_SIZE
        preliminary_aucpr_list_alpha = [[] for i in range(9)]
        preliminary_aucroc_list_alpha = [[] for i in range(9)]
        preliminary_overall_aucpr_list = []
        preliminary_overall_aucroc_list = []

        # as the Slim-tree is a bit sensitive to the insertion order of the input dataset, 10 random permutations
        # of the dataset instances are used to average the performance of C-AllOut
        for dataset_shuffle in np.arange(0, 10, 1):
            data_input = np.loadtxt(benchmark_path + benchmark_name + '.txt', dtype=str, delimiter=';')
            data_input_labels = np.loadtxt(benchmark_path + y_benchmark_name, dtype=str, delimiter=';')
            X, y = shuffle(data_input, data_input_labels)
            np.savetxt(benchmark_path + str(dataset_shuffle) + benchmark_name + '.txt', X, fmt="%s", delimiter=';')
            np.savetxt(benchmark_path + str(dataset_shuffle) + y_benchmark_name, y, fmt="%s", delimiter=';')
            data_input = np.loadtxt(benchmark_path + str(dataset_shuffle) + benchmark_name + '.txt', dtype=str, delimiter=';')

            ## initialize list of 9 independent lists
            avg_aucpr_list_alpha = [[] for i in range(9)]
            avg_aucroc_list_alpha = [[] for i in range(9)]
            avg_overall_aucpr_list = []
            avg_overall_aucroc_list = []

            # in each iteration, the input dataset is sorted using the overall ranking, so that outliers become
            # the instances inserted for last. 10 iterations is the recommended amount
            for iteration in range(10):
                all_score_lists = []

                np.savetxt(benchmark_path + str(dataset_shuffle) + 'sequential_' + benchmark_name + '.txt', data_input, fmt="%s", delimiter=';')

                slimtree_output_file = 'sequential_' + str(dataset_shuffle) + benchmark_name + '_' + str(pg) + '.txt'
                cmd = './callout ' + str(pg) + ' ' + slimtree_output_file + ' ' + benchmark_path + str(dataset_shuffle) + 'sequential_' + benchmark_name + '.txt ' + benchmark_data_lines
                os.system(cmd)

                raw_scores = np.loadtxt("scores/scores_" + slimtree_output_file, dtype=float, delimiter=';')
                raw_scores = raw_scores[raw_scores[:, 0].argsort()]
                ids = np.array(raw_scores[:, 0])

                # the distance from every leaf node representative to its closest object is used as a scaling factor so that
                # the rankings can handle data clusters with different densities
                closest_to_representative = raw_scores[:, 2]
                closest_to_representative_shifted = [x + 1.0 for x in closest_to_representative]

                # work with the nearest neighbor of all points to use later in global and collective rankings
                min_dist_to_neighborhood = raw_scores[:, 3]
                min_dist_to_neighborhood = np.column_stack((ids, min_dist_to_neighborhood))
                min_dist_to_neighborhood_normalized = np.copy(min_dist_to_neighborhood)
                min_dist_to_neighborhood_normalized[:, 1] = min_dist_to_neighborhood[:, 1] / closest_to_representative_shifted
                min_dist_to_neighborhood_normalized_shifted = np.copy(min_dist_to_neighborhood_normalized)
                min_dist_to_neighborhood_normalized_shifted[:, 1] = min_dist_to_neighborhood_normalized[:, 1] + 1

                # distance to every neighbor is used in the filtering step of the local ranking
                NNDistances = []
                with open("scores/distToNeighborhood_" + slimtree_output_file) as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        NNDistances.append(list(map(float, line.split(';'))))

                # distance to the neighborhood and leaf radii need to be normalized so that the filtering works
                # with data clusters of different densities
                NNDistances_normalized = np.copy(NNDistances)
                NNDistances_normalized = [[x[0]] + list(np.array(x[1:]) / (1.0 + np.array(closest_to_representative_shifted[count]))) for count, x in enumerate(NNDistances_normalized)]

                # work with the leaf radii of all leaf nodes to use later in the filtering step of the local ranking
                leaf_radius = raw_scores[:, 4]
                leaf_radius[leaf_radius == -1.0] = leaf_radius.max()
                leaf_radius = np.column_stack((ids, leaf_radius))
                normalized_median_leaves = np.copy(leaf_radius)
                normalized_median_leaves[:, 1] = leaf_radius[:, 1] / closest_to_representative_shifted
                normalized_median_leaves_sorted = normalized_median_leaves[normalized_median_leaves[:, 1].argsort()[::-1]]
                # np.unique sorts ascendent, so we have to invert the list
                normalized_leaf_radius_unique = np.unique(normalized_median_leaves_sorted[:, 1])[::-1]
                normalized_leaf_radius_unique = np.column_stack((np.arange(0, len(normalized_leaf_radius_unique)), normalized_leaf_radius_unique))

                ################################

                # overall ranking as provided by the Slim-tree itself
                overall_ranking = raw_scores[:, 1]
                overall_ranking[overall_ranking == -1.0] = overall_ranking.max()
                overall_ranking = np.column_stack((ids, overall_ranking))
                overall_ranking_sorted = np.copy(overall_ranking)
                overall_ranking_sorted = overall_ranking_sorted[overall_ranking_sorted[:, 1].argsort()[::-1]]

                # global ranking is the overall ranking times the distance to the nearest neighbor normalized
                global_ranking = np.copy(overall_ranking)
                global_ranking[:, 1] = global_ranking[:, 1] * min_dist_to_neighborhood_normalized[:, 1]
                global_ranking_sorted = np.copy(global_ranking)
                global_ranking_sorted = global_ranking_sorted[global_ranking_sorted[:, 1].argsort()[::-1]]
                all_score_lists.append(global_ranking_sorted)

                # Kneedle algorithm is used to select the expected largest leaf radius which has only inlier points
                normalized_leaf_size_kn = KneeLocator(np.arange(0, len(normalized_leaf_radius_unique), 1), sorted(normalized_leaf_radius_unique[:, 1]), curve='convex', direction='increasing', S=10)
                normalized_selected_leaf_size = normalized_leaf_radius_unique[normalized_leaf_size_kn.knee, 1]

                # proceed to discard every point which has any neighbors inside the selected leaf radius.
                # This will, theoretically, leave only global and local outliers in the global ranking.
                # The reversed global ranking has local outliers at the top
                if normalized_leaf_size_kn.knee == None:
                    local_ranking_sorted = np.copy(global_ranking_sorted)
                else:
                    objects_to_discard = [x[0] for x in NNDistances_normalized if any(y < normalized_selected_leaf_size for y in x[1:])]
                    local_ranking = np.copy(global_ranking_sorted)
                    local_ranking = np.array([x for x in local_ranking if x[0] not in objects_to_discard])
                    if len(local_ranking) == 0:
                        local_ranking_sorted = np.copy(global_ranking_sorted)
                    else:
                        local_ranking_sorted = np.copy(local_ranking)
                        local_ranking_complement_sorted = np.copy(global_ranking_sorted)
                        local_ranking_complement_sorted = np.array([x for x in local_ranking_complement_sorted if x[0] in objects_to_discard])
                        local_ranking_sorted = np.vstack((local_ranking_sorted, local_ranking_complement_sorted))
                all_score_lists.append(local_ranking_sorted)

                # collective ranking is the overall ranking divided by the distance to the nearest neighbor normalized and shifted
                collective_ranking = np.copy(overall_ranking)
                collective_ranking[:, 1] = overall_ranking[:, 1] / min_dist_to_neighborhood_normalized_shifted[:, 1]
                collective_ranking_sorted = collective_ranking[collective_ranking[:, 1].argsort()[::-1]]
                all_score_lists.append(collective_ranking_sorted)

                ################################

                # save AUCROC and AUCPR performance measures for each ranking
                saveScores(overall_ranking_sorted, raw_scores.shape[0], test_y_outlier,  test_y_not_outlier, avg_overall_aucpr_list, avg_overall_aucroc_list)
                for score_count, score in enumerate(all_score_lists):
                    saveScores(score, raw_scores.shape[0], test_y_local_outlier,  test_y_not_outlier + test_y_global_outlier + test_y_collective_outlier,
                               avg_aucpr_list_alpha[3 * score_count], avg_aucroc_list_alpha[3 * score_count])
                    saveScores(score, raw_scores.shape[0], test_y_global_outlier,  test_y_not_outlier + test_y_local_outlier + test_y_collective_outlier,
                               avg_aucpr_list_alpha[3 * score_count + 1], avg_aucroc_list_alpha[3 * score_count + 1])
                    saveScores(score, raw_scores.shape[0], test_y_collective_outlier,  test_y_not_outlier + test_y_local_outlier + test_y_global_outlier,
                               avg_aucpr_list_alpha[3 * score_count + 2], avg_aucroc_list_alpha[3 * score_count + 2])

                ################################

                # use the overall ranking to sort the input dataset for the next iteration, so that
                # outliers become the last points to be inserted
                data_input_float = list(map(float, data_input[:, 0]))
                data_input_sorted_indexes = [np.where(data_input_float == x) for x in overall_ranking_sorted[:, 0][::-1]]
                data_input_sorted_indexes = np.concatenate(np.concatenate(data_input_sorted_indexes))
                previous_data_input = np.copy(data_input)
                data_input = data_input[data_input_sorted_indexes]
                if np.array_equal(previous_data_input, data_input):
                    break

            # the following lines will take averages of the AUCROC and AUCPR measures per the iterations and random initializations
            for alpha_i in np.arange(0, 9, 1):
                preliminary_aucpr_list_alpha[alpha_i].append(avg_aucpr_list_alpha[alpha_i])
                preliminary_aucroc_list_alpha[alpha_i].append(avg_aucroc_list_alpha[alpha_i])
            preliminary_overall_aucpr_list.append(avg_overall_aucpr_list)
            preliminary_overall_aucroc_list.append(avg_overall_aucroc_list)

        preliminary_aucpr_list_alpha = np.array(preliminary_aucpr_list_alpha)
        preliminary_aucroc_list_alpha = np.array(preliminary_aucroc_list_alpha)
        preliminary_overall_aucpr_list = np.array(preliminary_overall_aucpr_list)
        preliminary_overall_aucroc_list = np.array(preliminary_overall_aucroc_list)

        rand_aucpr_list_alpha = [[] for i in range(9)]
        rand_aucroc_list_alpha = [[] for i in range(9)]
        rand_overall_aucpr_list = []
        rand_overall_aucroc_list = []

        for rand_count in range(len(avg_overall_aucpr_list)):
            for alpha_i in np.arange(0, 9, 1):
                rand_aucpr_list_alpha[alpha_i].append(np.mean(preliminary_aucpr_list_alpha[alpha_i][:, rand_count]))
                rand_aucroc_list_alpha[alpha_i].append(np.mean(preliminary_aucroc_list_alpha[alpha_i][:, rand_count]))
            rand_overall_aucpr_list.append(np.mean(preliminary_overall_aucpr_list[:, rand_count]))
            rand_overall_aucroc_list.append(np.mean(preliminary_overall_aucroc_list[:, rand_count]))
            
        for alpha_i in np.arange(0, 9, 1):
            rand_aucpr_list_alpha[alpha_i] = np.append(rand_aucpr_list_alpha[alpha_i], np.mean(rand_aucpr_list_alpha[alpha_i]))
            rand_aucroc_list_alpha[alpha_i] = np.append(rand_aucroc_list_alpha[alpha_i], np.mean(rand_aucroc_list_alpha[alpha_i]))
        rand_overall_aucpr_list = np.append(rand_overall_aucpr_list, np.mean(rand_overall_aucpr_list))
        rand_overall_aucroc_list = np.append(rand_overall_aucroc_list, np.mean(rand_overall_aucroc_list))

        currpage_aucpr_list = []
        currpage_aucroc_list = []

        for elem in rand_aucpr_list_alpha:
            currpage_aucpr_list.append(elem)
        for elem in rand_aucroc_list_alpha:   
            currpage_aucroc_list.append(elem)
        currpage_aucpr_list.append(rand_overall_aucpr_list)
        currpage_aucroc_list.append(rand_overall_aucroc_list)

        currpage_aucpr_list = np.array(currpage_aucpr_list)
        currpage_aucroc_list = np.array(currpage_aucroc_list)

        # save the AUCPR and AUCROC measures for each iteration
        np.savetxt(("./progression_stats/" + benchmark_name + "_" + str(pg) + "_pxr.csv"), currpage_aucpr_list, fmt='%s', delimiter=',')
        np.savetxt(("./progression_stats/" + benchmark_name + "_" + str(pg) + "_roc.csv"), currpage_aucroc_list, fmt='%s', delimiter=',')

        # select only the AUCPR and AUCROC measures for the 10th iteration to be the final measurements
        for alpha_i in np.arange(0, 9, 1):
            aucpr_list_alpha[alpha_i].append(rand_aucpr_list_alpha[alpha_i][-2])
            aucroc_list_alpha[alpha_i].append(rand_aucroc_list_alpha[alpha_i][-2])
        overall_aucpr_list.append(rand_overall_aucpr_list[-2])
        overall_aucroc_list.append(rand_overall_aucroc_list[-2])
        
    #############################
    
    for elem in aucpr_list_alpha:
        aucpr_list.append(elem)
    for elem in aucroc_list_alpha:   
        aucroc_list.append(elem)
    aucpr_list.append(overall_aucpr_list)
    aucroc_list.append(overall_aucroc_list)

    aucpr_list = np.array(aucpr_list)
    aucroc_list = np.array(aucroc_list)

    np.savetxt(("./stats/" + benchmark_name + "_pxr.csv"), aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "_roc.csv"), aucroc_list, fmt='%s', delimiter=',')
