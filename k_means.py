from sklearn.cluster import KMeans
import numpy as np
import data_service
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def print_accuracy(y_labels, y_labels_clusterred, n_clusters):

    cluster_labels=[[] for i in range(n_clusters)]
    for i, j in enumerate(y_labels_clusterred):
        cluster_labels[j].append(y_labels[i])

    total_number_most_common_for_cluster = 0
    for cluster in cluster_labels:
        if len(cluster) == 0:
            continue
        unique_labels, unique_labels_counts = np.unique(np.asarray(cluster), return_counts=True)
        # print(unique_labels)
        # print(unique_labels_counts)
        label_most_common_for_this_cluster = unique_labels[np.argmax(unique_labels_counts)]
        #assuming most of records in the cluster are from the same class
        num_records_for_most_common_label = np.max(unique_labels_counts)
        total_number_most_common_for_cluster += num_records_for_most_common_label
        print('Class {0}, Accuracy: {1}, Records for most common class: {2}, All Records: {3}, '
              .format(label_most_common_for_this_cluster, num_records_for_most_common_label/(1.0 * len(cluster)),
                num_records_for_most_common_label, len(cluster)))
        print('ClassesAndCounts: {0}: '.format(dict(zip(unique_labels, unique_labels_counts))))
        print('-------------------------------------------------')

    print('Overall Accuracy: {0}'.format(total_number_most_common_for_cluster / (1.0 * y_labels.shape[0])))
    print('-------------------------------------')


def run_k_means(x_train, x_test, y_train, y_test):

    n_clusters = np.unique(y_train).shape[0]

    k_means = KMeans(n_clusters=n_clusters, random_state=None).fit(x_train)

    print("TRAINING:")
    print_accuracy(y_train, k_means.labels_, n_clusters)
    print("********** END TRAINING ************\n\n")

    print("TESTING:")
    test_prediction = k_means.predict(x_test)
    n_clusters = np.unique(y_test).shape[0]
    print_accuracy(y_test, test_prediction, n_clusters)
    print("********** END TESTING ************")


scale_data = True
transform_data = True
random_slice = 25000
random_seed = None
test_size = 0.5

dataset = 'breast_cancer'
dataset = 'kdd'
#dataset = 'covtype'


x_train, x_test, y_train, y_test = data_service.\
    load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                        random_seed=random_seed, dataset=dataset, test_size=test_size)


run_k_means(x_train, x_test, y_train, y_test)

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
