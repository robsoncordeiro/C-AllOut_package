import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# list of colors for curves
possible_colors = ['silver', 'lightcoral', 'brown', 'red', 'darkorange', 'gold', 'lawngreen', 'mediumseagreen', 'darkgreen', 'turquoise', 'cyan', 'blue', 'blueviolet', 'magenta', 'pink', 'tan', 'gainsboro', 'wheat', 'teal', 'steelblue', 'black']

directory = "progression_stats/"

relative_change_overall = []
overall_progression_list = []

# get sorted list of all the files in directory
files = sorted(os.listdir(directory))
for filename in files:
    # select only files related to AUCPR
    if "pxr" in filename:
        progression = np.loadtxt("progression_stats/" + filename, delimiter=',', dtype=float)

        overall_progression = progression[2, :-1]
        overall_progression_list.append(overall_progression)

        # get the relative gain for each iteration in comparison with the first iteration
        starting_iteration_value = overall_progression[0]
        relative_change_overall.append([(x - starting_iteration_value) / starting_iteration_value for x in overall_progression])

# take the average gain for all datasets
avg_relative_change_overall = [np.mean(np.array(relative_change_overall)[:, count]) for count in range(30)]
relative_change_overall.append(avg_relative_change_overall)
relative_change_overall = np.array(relative_change_overall)

# create graph with the average curve
legend_label = ["Average Curve Across Datasets"]
plt.rcParams.update({'font.size': 15})
for count in range(len(relative_change_overall)):
    if count < 20:
        pass
        # plt.plot(np.arange(1, 31, 1), relative_change_overall[count, :] * 100, marker='o', color=possible_colors[count], alpha=0.4, markersize=2)
    else:
        plt.plot(np.arange(1, 31, 1), relative_change_overall[count, :] * 100, marker='o', color=possible_colors[count], alpha=1, linewidth=3)
plt.axhline(y=0, color='r', linestyle='--')
plt.subplots_adjust(top=0.97, right=0.99, left=0.16)
plt.ylim(-50, 100)
plt.xlabel('Iteration')
plt.ylabel("AUCPR % Gain Against First Iteration")
lines = [Line2D([0], [0], color=c, linewidth=2, linestyle='-') for c in [possible_colors[-1]]]
plt.legend(lines, legend_label, loc='upper center', ncol=4, prop={'size': 13}, handletextpad=0.2, labelspacing=0.2, columnspacing=0.3)
plt.savefig('AUCPR_overallScoreRelChange.pdf')
plt.show()
plt.close('all')
    