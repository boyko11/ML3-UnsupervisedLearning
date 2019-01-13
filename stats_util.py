import numpy as np


def generate_stats(y_labels, y_labels_clustered, n_clusters, x_data):
    labels_prior_to_clustering = np.histogram(y_labels, bins=np.arange(n_clusters + 1))[0]

    class_accuracies = np.zeros(n_clusters)
    clusters_per_class_count = np.zeros(n_clusters)
    instances_per_clusters_per_class_count = np.zeros(n_clusters)
    majority_class_instances_per_clusters_per_class_count = np.zeros(n_clusters)
    cluster_labels = [[] for i in range(n_clusters)]
    custer_train_indices = [[] for i in range(n_clusters)]
    for i, j in enumerate(y_labels_clustered):
        cluster_labels[j].append(y_labels[i])
        custer_train_indices[j].append(i)

    total_number_most_common_for_cluster = 0
    for cluster_index, cluster in enumerate(cluster_labels):
        if len(cluster) == 0:
            continue
        unique_labels, unique_labels_counts = np.unique(np.asarray(cluster), return_counts=True)
        label_most_common_for_this_cluster = unique_labels[np.argmax(unique_labels_counts)]
        cluster_train_indices_np = np.asarray(custer_train_indices[cluster_index])
        cluster_y_labels = y_labels[cluster_train_indices_np]
        cluster_x_train_records = x_data[cluster_train_indices_np, :]
        label_most_common_train_indices = np.where(cluster_y_labels[cluster_y_labels == label_most_common_for_this_cluster])[0]
        cluster_x_train_records_label_most_common = cluster_x_train_records[label_most_common_train_indices, :]
        # if label_most_common_for_this_cluster == 11:
        #     print("Normal stats: ")
        #     print_attribute_stats(cluster_x_train_records_label_most_common)
        #     print("Boyko Todorov wrote this")
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

def print_attribute_stats(numpy_array):

    min_stats = np.min(numpy_array, axis=0)
    max_stats = np.max(numpy_array, axis=0)
    mean_stats = np.mean(numpy_array, axis=0)
    std_stats = np.std(numpy_array, axis=0)
    print("MIN across attributes: ")
    print(min_stats)
    print("MAX across attributes: ")
    print(max_stats)
    print("MEAN across attributes: ")
    print(mean_stats)
    print("STD across attributes: ")
    print(std_stats)
    print('------------------------------')