from sklearn.cluster import KMeans
import numpy as np
import data_service
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys
from tabulate import tabulate


def update_avg_stats(class_stats, stats_avg, run_index):

    for this_index, this_class_stats in enumerate(class_stats):

        this_class_stats = np.asarray(this_class_stats)
        if run_index > 0:

            if np.count_nonzero(this_class_stats[1:5]) == 0:
                #if all zeros for this test run, replace with average from previous runs
                this_class_stats[1:5] = stats_avg[this_index, 1:5] / run_index
            if np.count_nonzero(stats_avg[this_index, 1:5]) == 0:
                stats_avg[this_index, 1:5] = this_class_stats[1:5]
        stats_avg[this_index, :] += this_class_stats
        if this_index == 15:
            print('Class15: ', run_index, stats_avg[this_index, :])
        if this_index == 17:
            print('Class17: ', run_index, stats_avg[this_index, :])

    return stats_avg

def generate_stats(y_labels, y_labels_clustered, n_clusters):
    labels_prior_to_clustering = np.histogram(y_labels, bins=np.arange(n_clusters + 1))[0]

    class_accuracies = np.zeros(n_clusters)
    clusters_per_class_count = np.zeros(n_clusters)
    instances_per_clusters_per_class_count = np.zeros(n_clusters)
    majority_class_instances_per_clusters_per_class_count = np.zeros(n_clusters)
    cluster_labels = [[] for i in range(n_clusters)]
    for i, j in enumerate(y_labels_clustered):
        cluster_labels[j].append(y_labels[i])

    total_number_most_common_for_cluster = 0
    for cluster in cluster_labels:
        if len(cluster) == 0:
            continue
        unique_labels, unique_labels_counts = np.unique(np.asarray(cluster), return_counts=True)
        label_most_common_for_this_cluster = unique_labels[np.argmax(unique_labels_counts)]
        # assuming most of records in the cluster are from the same class
        num_records_for_most_common_label = np.max(unique_labels_counts)
        total_number_most_common_for_cluster += num_records_for_most_common_label
        accuracy = num_records_for_most_common_label / (1.0 * len(cluster))
        print('Class {0}, Accuracy: {1}, Records for most common class: {2}, All Records: {3}, '
              .format(label_most_common_for_this_cluster, accuracy, num_records_for_most_common_label, len(cluster)))
        print('ClassesAndCounts: {0}: '.format(dict(zip(unique_labels, unique_labels_counts))))
        print('-------------------------------------------------')
        class_accuracies[label_most_common_for_this_cluster] += accuracy
        clusters_per_class_count[label_most_common_for_this_cluster] += 1
        majority_class_instances_per_clusters_per_class_count[label_most_common_for_this_cluster] += \
            num_records_for_most_common_label
        instances_per_clusters_per_class_count[label_most_common_for_this_cluster] += len(cluster)

    clusters_per_class_count_for_calculation = clusters_per_class_count.copy()
    clusters_per_class_count_for_calculation[clusters_per_class_count_for_calculation == 0] = 1
    class_accuracies = class_accuracies / clusters_per_class_count_for_calculation
    majority_class_instances_per_clusters_per_class_count = majority_class_instances_per_clusters_per_class_count / \
        clusters_per_class_count_for_calculation
    instances_per_clusters_per_class_count = instances_per_clusters_per_class_count / \
         clusters_per_class_count_for_calculation

    overall_accuracy = total_number_most_common_for_cluster / (1.0 * y_labels.shape[0])
    return overall_accuracy, list(zip(range(n_clusters), class_accuracies, clusters_per_class_count,
                                       majority_class_instances_per_clusters_per_class_count,
                                       instances_per_clusters_per_class_count, labels_prior_to_clustering))


def run_k_means(x_train, x_test, y_train, y_test, n_clusters):
    unique_labels, unique_labels_counts = np.unique(y_train, return_counts=True)
    unique_labels_num = unique_labels.shape[0]
    print("Data Has {0} unique labes".format(unique_labels_num))

    k_means = KMeans(n_clusters=n_clusters, random_state=None).fit(x_train)

    print("TRAINING:")
    overall_train_accuracy, train_stats_per_class = generate_stats(y_train, k_means.labels_, n_clusters)
    print("********** END TRAINING ************\n\n")

    print("TESTING:")
    test_prediction = k_means.predict(x_test)
    overall_test_accuracy, test_stats_per_class = generate_stats(y_test, test_prediction, n_clusters)

    print("********** END TESTING ************")

    return overall_train_accuracy, train_stats_per_class, overall_test_accuracy, test_stats_per_class


np.set_printoptions(suppress=True)
if len(sys.argv) < 2 or (sys.argv[1] != 'breast_cancer' and sys.argv[1] != 'kdd'):
    print("Usage: python k_means.py breast_cancer")
    print("or")
    print("Usage: python k_means.py kdd")
    exit()

dataset = sys.argv[1]

scale_data = True
transform_data = False
random_slice = None
random_seed = None
test_size = 0.5
num_unique_classes = 2
num_dimensions = 30

if dataset == 'kdd':
    transform_data = True
    random_slice = None
    test_size = 0.5
    num_unique_classes = 23

num_test_runs = 10
train_scores = []
test_scores = []
train_stats_avg = np.zeros((num_unique_classes, 6))
test_stats_avg = np.zeros((num_unique_classes, 6))
for test_run_index in range(num_test_runs):
    x_train, x_test, y_train, y_test = data_service. \
        load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                            random_seed=random_seed, dataset=dataset, test_size=test_size)

    train_score, class_train_stats, test_score, class_test_stats = \
        run_k_means(x_train, x_test, y_train, y_test, num_unique_classes)

    train_scores.append(train_score)
    test_scores.append(test_score)

    train_stats_avg = update_avg_stats(class_train_stats, train_stats_avg, test_run_index)
    test_stats_avg = update_avg_stats(class_test_stats, test_stats_avg, test_run_index)


avg_train_score = np.mean(np.asarray(train_scores))
avg_test_score = np.mean(np.asarray(test_scores))
print("Pure KMeans Results:")
print("Average Training Score: {0}".format(avg_train_score))
print("Average Testing Score: {0}".format(avg_test_score))

headers = ["Class", "ClusterScore", "Score", "NumOfClusters", "NumMajorityInCluster", "AllRecordsInCluster", "TrainRecords"]


train_stats_avg = np.insert(train_stats_avg, 2, 0, axis=1)
test_stats_avg = np.insert(test_stats_avg, 2, 0, axis=1)
print(train_stats_avg.shape)
print(train_stats_avg.shape)
train_stats_avg[:, 2] = (train_stats_avg[:, 3] * train_stats_avg[:, 4])/ train_stats_avg[:, 6]
test_stats_avg[:, 2] = (test_stats_avg[:, 3] * test_stats_avg[:, 4])/ test_stats_avg[:, 6]

print("Avg per Class Training Stats:")
train_stats_table = tabulate(train_stats_avg / num_test_runs, headers, tablefmt="simple")
print(train_stats_table)

print("Avg per Class Testing Stats:")
test_stats_table = tabulate(test_stats_avg / num_test_runs, headers, tablefmt="simple")
print(test_stats_table)

exit()

print("Applying PCA...")

pca = PCA(n_components=10)
x_train_PCA = pca.fit_transform(x_train.copy())
x_test_PCA = pca.transform(x_test.copy())

run_k_means(x_train_PCA, x_test_PCA, y_train, y_test)

print("Applying ICA...")

fastICA = FastICA(n_components=3, random_state=0)
x_train_ICA = fastICA.fit_transform(x_train.copy())
x_test_ICA = fastICA.transform(x_test.copy())

run_k_means(x_train_ICA, x_test_ICA, y_train, y_test)

print("Applying RCA...")

rca = GaussianRandomProjection(n_components=26)
x_train_RCA = rca.fit_transform(x_train.copy())
x_test_RCA = rca.transform(x_test.copy())

run_k_means(x_train_RCA, x_test_RCA, y_train, y_test)

print("Applying LDA...")

rca = LinearDiscriminantAnalysis(n_components=1)
x_train_LDA = rca.fit_transform(x_train.copy(), y_train)
x_test_LDA = rca.transform(x_test.copy())

run_k_means(x_train_LDA, x_test_LDA, y_train, y_test)
