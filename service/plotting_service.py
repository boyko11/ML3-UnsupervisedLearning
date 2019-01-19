import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

markers = ["^", "o", "v", "<", ">", "1", "2", "3", "4", "*", "8", ".", "s",  "p", "P", "h", "H", "+", "x", "X", "D",
           "d", "|"]

# markers = ["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$", "$12$", "$13$", "$14$", "$15$", "$16$", "$17$", "$18$", "$19$", "$20$", "$21$", "$22$", "$23$" ]
colors = ["b", "g", "r", "c", "m", "y", "k", "b", "g", "C0", "r", "C1", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k", "b", "g", "r", "c", "m", "y", "k"]

def plot_scores_per_pcs(accuracy_scores_per_pcs, accuracy_scores_per_ics, accuracy_scores_per_rcs,
                        accuracy_scores_per_ldacs, accuracy_scores_per_pcs_plus_c,  accuracy_scores_per_ics_plus_c,
                        accuracy_scores_per_rcs_plus_c, accuracy_scores_per_ldacs_plus_c, original_accuracy, dataset):

    plt.figure()
    plt.title("{0} Accuracy Per Number of Components".format(dataset))
    plt.xlabel("Number of Components")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.plot(np.arange(1, len(accuracy_scores_per_pcs) + 1), accuracy_scores_per_pcs, color="b", label='PCA')
    plt.plot(np.arange(1, len(accuracy_scores_per_pcs_plus_c) + 1), accuracy_scores_per_pcs_plus_c, color="b", label='PCA+', linestyle=':')
    plt.plot(np.arange(1, len(accuracy_scores_per_ics) + 1), accuracy_scores_per_ics, color="C0", label='ICA')
    plt.plot(np.arange(1, len(accuracy_scores_per_ics_plus_c) + 1), accuracy_scores_per_ics_plus_c, color="C0", label='ICA+', linestyle=':')
    plt.plot(np.arange(1, len(accuracy_scores_per_rcs) + 1), accuracy_scores_per_rcs, color="r", label='RCA')
    plt.plot(np.arange(1, len(accuracy_scores_per_rcs_plus_c) + 1), accuracy_scores_per_rcs_plus_c, color="r", label='RCA+', linestyle=':')
    plt.plot(np.arange(1, len(accuracy_scores_per_ldacs) + 1), accuracy_scores_per_ldacs, color="C1", label='LDA')
    plt.plot(np.arange(1, len(accuracy_scores_per_ldacs_plus_c) + 1), accuracy_scores_per_ldacs_plus_c, color="C1", label='LDA+', linestyle=':')
    plt.axhline(y=original_accuracy, color='g', linestyle='-', label='Mean Non-Reduced Score')

    # plt.legend(loc="best")
    plt.ylim(0.96, 1.00)
    #plt.show()
    plt.savefig('NeuralNetReduceAndCluster-{0}.png'.format(dataset))

def plot_eigenvalues(eigenvalues_breast_cancer, eigenvalues_kdd ):

    plt.figure()
    plt.title("Eigenvalues")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")

    plt.grid()

    plt.plot(np.arange(1, len(eigenvalues_breast_cancer) + 1), eigenvalues_breast_cancer, color="C0", label='BreastCancer')
    plt.plot(np.arange(1, len(eigenvalues_kdd) + 1), eigenvalues_kdd, color="C1", label='KDD')

    plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    plt.show()

def plot_eignevalues_barchart(eigenvalues_breast_cancer, eigenvalues_kdd):

    n_groups = eigenvalues_kdd.shape[0]

    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups) + 1
    bar_width = 0.4
    opacity = 0.8

    plt.bar(np.arange(eigenvalues_breast_cancer.shape[0]) + 1, eigenvalues_breast_cancer, bar_width,
                     alpha=opacity,
                     color='C0',
                     label='Breast Cancer')

    plt.bar(index + bar_width, eigenvalues_kdd, bar_width,
                     alpha=opacity,
                     color='C1',
                     label='KDD')

    plt.xlabel('Component')
    plt.ylabel('Eignevalue')
    plt.title('Eigenvalues')
    #plt.xticks(index + bar_width, index + 1)
    plt.legend(loc='best')
    # plt.grid()
    #
    # plt.tight_layout()
    #plt.show()
    plt.savefig('Eigenvalues.png')


def plot3D_scatter(feature_data, labels, reduction_algo, dataset_name):
    # https://matplotlib.org/gallery/mplot3d/scatter3d.html
    # quoted source's; if you use this file quote me as Boyko Todorov
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
    #plt.show()
    plt.savefig('Reduce3D-{0}-{1}.png'.format(reduction_algo, dataset_name))

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
    #plt.show()
    plt.savefig('Reduce2D-{0}-{1}.png'.format(reduction_algo, dataset_name))

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
    #plt.show()
    plt.savefig('Reduce1D-{0}-{1}.png'.format(reduction_algo, dataset_name))

def plot_projection_losses(projection_loss, dataset, algo):

    plt.figure()
    plt.title("{0} {1} Projection Loss Per Number of Components".format(algo, dataset))
    plt.xlabel("Number of Components")
    plt.ylabel("Projection Loss")

    plt.grid()

    plt.plot(np.arange(1, len(projection_loss) + 1), projection_loss, color="r")

    plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    #plt.show()
    plt.savefig('ProjectionLoss-{0}-{1}.png'.format(algo, dataset))


def plot_std_per_component(std, dataset, algo):

    plt.figure()
    plt.title("{0} {1} STD Per Number of Components".format(algo, dataset))
    plt.xlabel("Number of Components")
    plt.ylabel("STD")

    plt.grid()

    plt.plot(np.arange(1, len(std) + 1), std, color="r")

    plt.legend(loc="best")
    #plt.ylim(0.96, 1.00)
    #plt.show()
    plt.savefig('ProjectionSTD-{0}-{1}.png'.format(algo, dataset))


def plot_component_stat_line(stat, stat_name, title, dataset):

    plt.figure()
    plt.title(title)
    plt.xlabel("Number of Components")
    plt.ylabel(stat_name)

    plt.grid()

    plt.plot(np.arange(1, len(stat) + 1), stat)

    plt.legend(loc="best")
    #plt.show()
    plt.savefig('Kurtosis-{0}.png'.format(dataset))







