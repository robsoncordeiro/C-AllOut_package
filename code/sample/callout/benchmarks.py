import datasets_list
import os
import numpy as np
from experiments_utils import saveScores
from sklearn.utils import shuffle
from kneed import KneeLocator

# the constant header size in bytes for a Slim-tree node. 
# The header of a node only stores metadata about the node
NODE_HEADER_SIZE = 10

for [benchmark_path, benchmark_name, y_benchmark_name, benchmark_data_lines, benchmark_data_columns, benchmark_object_byte_size] in datasets_list.datasets_benchmark:
    # Separate the labels in two lists, one with all the inliers (y = 0) and one with all the outliers (y = 1)
    f=open(benchmark_path + y_benchmark_name, "r")
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

                ################################

                # overall ranking as provided by the Slim-tree itself
                overall_ranking = raw_scores[:, 1]
                overall_ranking = np.column_stack((ids, overall_ranking))
                overall_ranking_sorted = np.copy(overall_ranking)
                overall_ranking_sorted = overall_ranking_sorted[overall_ranking_sorted[:, 1].argsort()[::-1]]

                ################################

                # save AUCROC and AUCPR performance measures for the overall ranking
                saveScores(overall_ranking_sorted, raw_scores.shape[0], test_y_outlier, test_y_not_outlier, avg_overall_aucpr_list, avg_overall_aucroc_list)

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
            preliminary_overall_aucpr_list.append(avg_overall_aucpr_list)
            preliminary_overall_aucroc_list.append(avg_overall_aucroc_list)

        preliminary_overall_aucpr_list = np.array(preliminary_overall_aucpr_list)
        preliminary_overall_aucroc_list = np.array(preliminary_overall_aucroc_list)

        rand_overall_aucpr_list = []
        rand_overall_aucroc_list = []

        for rand_count in range(len(avg_overall_aucpr_list)):
            rand_overall_aucpr_list.append(np.mean(preliminary_overall_aucpr_list[:, rand_count]))
            rand_overall_aucroc_list.append(np.mean(preliminary_overall_aucroc_list[:, rand_count]))
            
        rand_overall_aucpr_list = np.append(rand_overall_aucpr_list, np.mean(rand_overall_aucpr_list))
        rand_overall_aucroc_list = np.append(rand_overall_aucroc_list, np.mean(rand_overall_aucroc_list))

        currpage_aucpr_list = []
        currpage_aucroc_list = []

        currpage_aucpr_list.append(rand_overall_aucpr_list)
        currpage_aucroc_list.append(rand_overall_aucroc_list)

        np.savetxt(("./progression_stats/" + benchmark_name + "_" + str(pg) + "_pxr.csv"), currpage_aucpr_list, fmt='%s', delimiter=',')
        np.savetxt(("./progression_stats/" + benchmark_name + "_" + str(pg) + "_roc.csv"), currpage_aucroc_list, fmt='%s', delimiter=',')

        overall_aucpr_list.append(rand_overall_aucpr_list[-2])
        overall_aucroc_list.append(rand_overall_aucroc_list[-2])
        
    #############################

    aucpr_list = np.array(overall_aucpr_list)
    aucroc_list = np.array(overall_aucroc_list)

    np.savetxt(("./stats/" + benchmark_name + "_pxr.csv"), aucpr_list, fmt='%s', delimiter=',')
    np.savetxt(("./stats/" + benchmark_name + "_roc.csv"), aucroc_list, fmt='%s', delimiter=',')
