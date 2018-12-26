import numpy as np
import matplotlib.pyplot as plt


def plot_scores_per_pcs(accuracy_scores_per_pcs, original_accuracy):

    plt.figure()
    plt.title("Accuracy Per Number of Principal Components")
    plt.xlabel("Number PCs")
    plt.ylabel("Accuracy")

    plt.grid()

    plt.plot(np.arange(1, len(accuracy_scores_per_pcs) + 1), accuracy_scores_per_pcs, color="b", label='Mean PCA Score')
    plt.axhline(y=original_accuracy, color='g', linestyle='-', label='Mean Non-PCA Score')

    plt.legend(loc="best")
    plt.show()
