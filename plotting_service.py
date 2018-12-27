import numpy as np
import matplotlib.pyplot as plt


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
