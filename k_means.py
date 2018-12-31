from sklearn.cluster import KMeans
import numpy as np
import data_service
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sys


def print_accuracy(y_labels, y_labels_clustered, n_clusters):

    cluster_labels=[[] for i in range(n_clusters)]
    for i, j in enumerate(y_labels_clustered):
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

    overall_accuracy = total_number_most_common_for_cluster / (1.0 * y_labels.shape[0])
    print('Overall Accuracy: {0}'.format(overall_accuracy))
    print('-------------------------------------')
    return overall_accuracy


def run_k_means(x_train, x_test, y_train, y_test, n_clusters):

    unique_labels, unique_labels_counts = np.unique(y_train, return_counts=True)
    unique_labels_num = unique_labels.shape[0]
    print("Data Has {0} unique labes".format(unique_labels_num))
    orig_ratio = np.histogram(y_train, bins=np.arange(n_clusters+1), density=True)[0]
    print("Orig Ratio: ")
    print(orig_ratio)
    print(np.histogram(y_train, bins=np.arange(n_clusters+1))[0])

    k_means = KMeans(n_clusters=n_clusters, random_state=None).fit(x_train)

    # print("Cluster centers: ")
    # print(k_means.cluster_centers_)

    print("TRAINING:")
    train_accuracy = print_accuracy(y_train, k_means.labels_, n_clusters)
    train_ratio = np.histogram(k_means.labels_, bins=np.arange(n_clusters+1), density=True)[0]
    print("Train Ratio: ")
    print(train_ratio)
    print("********** END TRAINING ************\n\n")

    print("TESTING:")
    test_prediction = k_means.predict(x_test)
    test_accuracy = print_accuracy(y_test, test_prediction, n_clusters)
    test_ratio = np.histogram(test_prediction, bins=np.arange(n_clusters+1), density=True)[0]
    print("Test Ratio: ")
    print(test_ratio)
    print("********** END TESTING ************")

    train_ratio.sort()
    test_ratio.sort()
    return train_accuracy, test_accuracy, orig_ratio, train_ratio, test_ratio


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
test_size = 0.2
num_unique_classes = 2
num_dimensions = 30

if dataset == 'kdd':
    transform_data = True
    random_slice = 25000
    test_size = 0.5
    num_unique_classes = 23


num_test_runs = 1
train_scores = []
test_scores = []
orig_ratios_total = np.zeros(num_unique_classes)
train_ratios_total = np.zeros(num_unique_classes)
test_ratios_total = np.zeros(num_unique_classes)
for test_run_index in range(num_test_runs):
    x_train, x_test, y_train, y_test = data_service.\
        load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                            random_seed=random_seed, dataset=dataset, test_size=test_size)

    train_score, test_score, orig_ratios, train_ratios, test_ratios = run_k_means(x_train, x_test, y_train, y_test, num_unique_classes)
    orig_ratios_total = np.add(orig_ratios_total, orig_ratios)
    train_ratios_total = np.add(train_ratios_total, train_ratios)
    test_ratios_total = np.add(test_ratios_total, test_ratios)
    train_scores.append(train_score)
    test_scores.append(test_score)

avg_train_score = np.mean(np.asarray(train_scores))
avg_test_score = np.mean(np.asarray(test_scores))
print("Pure KMeans Results:")
print("Average Training Score: {0}".format(avg_train_score))
print("Average Testing Score: {0}".format(avg_test_score))

print("Original classes ratios: {0}".format(orig_ratios_total/num_test_runs))
print("Train classes ratios: {0}".format(train_ratios_total/num_test_runs))
print("Test classes ratios: {0}".format(test_ratios_total/num_test_runs))

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
