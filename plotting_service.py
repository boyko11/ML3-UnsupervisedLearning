import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

markers = ["^", "o", "v", "<", ">", "1", "2", "3", "4", "*", "8", ".", "s",  "p", "P", "h", "H", "+", "x", "X", "D",
           "d", "|"]

# markers = ["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$", "$12$", "$13$", "$14$", "$15$", "$16$", "$17$", "$18$", "$19$", "$20$", "$21$", "$22$", "$23$" ]
colors = ["b", "g", "r", "c", "m", "y", "k", "b", "g", "C0", "r", "C1", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k"]

def plot_scores_per_pcs(accuracy_scores_per_pcs, original_accuracy, reduction_algo):

    plt.figure()
    plt.title("Accuracy Per Number of {0} Components".format(reduction_algo))
    plt.xlabel("Number of {0} Components".format((reduction_algo)))
    plt.ylabel("Accuracy")

    plt.grid()

    plt.plot(np.arange(1, len(accuracy_scores_per_pcs) + 1), accuracy_scores_per_pcs, color="b", label='Mean {0} Score'.format(reduction_algo))
    plt.axhline(y=original_accuracy, color='g', linestyle='-', label='Mean Non-{0} Score'.format(reduction_algo))

    plt.legend(loc="best")
    plt.show()


def plot3D_scatter(feature_data, labels, reduction_algo, dataset_name):
    # https://matplotlib.org/gallery/mplot3d/scatter3d.html
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for index, record in enumerate(feature_data):
    #     this_label = labels[index]
    #     ax.scatter(record[0], record[1], record[2], c=colors[this_label], marker=markers[this_label], label="Class{0}".format(this_label))
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        indices_for_this_label = np.where(labels == unique_label)[0]
        records_for_this_label = feature_data[indices_for_this_label, :]
        ax.scatter(records_for_this_label[:, 0], records_for_this_label[:, 1], records_for_this_label[:, 2],
                   c=colors[unique_label], marker=markers[unique_label], label="{0}".format(unique_label))

    ax.set_xlabel('C1')
    ax.set_ylabel('C2')
    ax.set_zlabel('C3')
    plt.title("{0} - {1}".format(reduction_algo, dataset_name))
    #plt.legend(loc='best', numpoints=1, ncol=2, fontsize=5, bbox_to_anchor=(0, 0))
    plt.show()

def plot2D_scatter(feature_data, labels, reduction_algo, dataset_name):
    # https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
    unique_labels = np.unique(labels)
    scatters_per_class = []
    classes = []
    for unique_label in unique_labels:
        indices_for_this_label = np.where(labels == unique_label)[0]
        this_class_scatter = plt.scatter(feature_data[indices_for_this_label, 0], feature_data[indices_for_this_label, 1],
                    c=colors[unique_label], marker=markers[unique_label])
        scatters_per_class.append(this_class_scatter)
        classes.append('{0}'.format(unique_label))

    plt.title("{0} - {1}".format(reduction_algo, dataset_name))
    plt.xlabel("C1")
    plt.ylabel("C2")
    plt.legend(tuple(scatters_per_class), tuple(classes), scatterpoints=1, loc='best', ncol=6, fontsize=5)
    plt.show()

def plot1D_scatter(feature_data, labels, reduction_algo, dataset_name):
    # https://matplotlib.org/gallery/shapes_and_collections/scatter.html#sphx-glr-gallery-shapes-and-collections-scatter-py
    unique_labels = np.unique(labels)
    for unique_label in unique_labels:
        indices_for_this_label = np.where(labels == unique_label)[0]
        plt.scatter(feature_data[indices_for_this_label, 0], np.zeros_like(feature_data[indices_for_this_label, 0]),
                    c=colors[unique_label], marker=markers[unique_label])

    plt.title("{0} - {1}".format(reduction_algo, dataset_name))
    plt.xlabel("Component 1")
    plt.legend(loc="best")
    plt.show()







