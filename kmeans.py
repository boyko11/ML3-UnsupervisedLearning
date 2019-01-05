from sklearn.cluster import KMeans
import stats_util
import numpy as np

def run(x_train, x_test, y_train, y_test, n_clusters):
    unique_labels, unique_labels_counts = np.unique(y_train, return_counts=True)
    unique_labels_num = unique_labels.shape[0]
    print("Data Has {0} unique labes".format(unique_labels_num))

    k_means = KMeans(n_clusters=n_clusters, random_state=None).fit(x_train)

    print("TRAINING:")
    overall_train_accuracy, train_stats_per_class = stats_util.generate_stats(y_train, k_means.labels_, n_clusters, x_train)
    print("********** END TRAINING ************\n\n")

    print("TESTING:")
    test_prediction = k_means.predict(x_test)
    overall_test_accuracy, test_stats_per_class = stats_util.generate_stats(y_test, test_prediction, n_clusters, x_test)

    print("********** END TESTING ************")

    return overall_train_accuracy, train_stats_per_class, overall_test_accuracy, test_stats_per_class