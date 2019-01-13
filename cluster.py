import numpy as np
import data_service
import sys
from tabulate import tabulate
import stats_util
import kmeans
import em
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import convert_to_binary_util


def apply_reduction_and_cluster_reduced_data(reduce_algo_name, n_components, cluster_algo, x_train, x_test, y_train,
                                             y_test, num_unique_classes, dataset_name):

    reduce_algo = None
    if reduce_algo_name == 'PCA':
        reduce_algo = PCA(n_components=n_components, whiten=True)
    elif reduce_algo_name == 'ICA':
        reduce_algo = FastICA(n_components=n_components, whiten=True)
    elif reduce_algo_name == 'RCA':
        reduce_algo = GaussianRandomProjection(n_components=n_components)
    elif reduce_algo_name == 'LDA':
        reduce_algo = LinearDiscriminantAnalysis(n_components=n_components)

    if reduce_algo_name == 'LDA':
        # if dataset_name == 'kdd':
        #     y_train_converted, y_test_converted = convert_to_binary_util.convert(y_train, y_test, 11)
        #     x_train_reduced = reduce_algo.fit_transform(x_train, y_train_converted)
        #     x_test_reduced = reduce_algo.transform(x_test)
        #     return cluster_algo.run(x_train_reduced, x_test_reduced, y_train_converted, y_test_converted, 2)
        # else:
        x_train_reduced = reduce_algo.fit_transform(x_train, y_train)
    else:
        x_train_reduced = reduce_algo.fit_transform(x_train)
    x_test_reduced = reduce_algo.transform(x_test)

    return cluster_algo.run(x_train_reduced, x_test_reduced, y_train, y_test, num_unique_classes)


def print_stats(train_scores, test_scores, train_stats_avg, test_stats_avg, num_test_runs):

    avg_train_score = np.mean(np.asarray(train_scores))
    avg_test_score = np.mean(np.asarray(test_scores))
    print("Average Training Score: {0}".format(avg_train_score))
    print("Average Testing Score: {0}".format(avg_test_score))

    headers = ["Class", "ClusterScore", "Score", "NumOfClusters", "NumMajorityInCluster", "AllRecordsInCluster",
               "TrainRecords"]

    train_stats_avg = np.insert(train_stats_avg, 2, 0, axis=1)
    test_stats_avg = np.insert(test_stats_avg, 2, 0, axis=1)
    train_stats_avg[:, 2] = (train_stats_avg[:, 3] * train_stats_avg[:, 4]) / train_stats_avg[:, 6]
    test_stats_avg[:, 2] = (test_stats_avg[:, 3] * test_stats_avg[:, 4]) / test_stats_avg[:, 6]

    print("Avg per Class Training Stats:")
    train_stats_table = tabulate(train_stats_avg / num_test_runs, headers, tablefmt="simple")
    print(train_stats_table)

    print("Avg per Class Testing Stats:")
    test_stats_table = tabulate(test_stats_avg / num_test_runs, headers, tablefmt="simple")
    print(test_stats_table)


np.set_printoptions(suppress=True, precision=3)
if len(sys.argv) < 4 or (sys.argv[1] not in ['kmeans', 'em']) \
        or (not sys.argv[2].isdigit()) or (sys.argv[3] not in ['breast_cancer', 'kdd']):
    print("Usage: python cluster.py <cluster-algo> <number-of-reduction-components> <dataset> <optional:binary(to convert kdd from multiclass to good-bad dataset")
    print("Examples:")
    print("Usage: python cluster.py kmeans 2 breast_cancer")
    print("or")
    print("Usage: python cluster.py kmeans 3 kdd")
    print("or")
    print("Usage: python cluster.py kmeans 3 kdd binary")
    print("or")
    print("Usage: python cluster.py em 2 breast_cancer")
    print("or")
    print("Usage: python cluster.py em 10 kdd")
    print("or")
    print("Usage: python cluster.py em 2 kdd binary")
    exit()

cluster_algo = kmeans
algo_to_run = sys.argv[1]
if algo_to_run == 'em':
    cluster_algo = em

n_components = int(sys.argv[2])

dataset = sys.argv[3]

scale_data = True
transform_data = False
random_slice = None
random_seed = None
test_size = 0.5
num_unique_classes = 2
num_dimensions = 30

reduce_kdd_to_binary = False
if dataset == 'kdd':
    transform_data = True
    random_slice = None
    test_size = 0.5
    num_unique_classes = 23
    if len(sys.argv) > 4:
        reduce_kdd_to_binary = sys.argv[4] == 'binary'
        if reduce_kdd_to_binary:
            num_unique_classes = 2

print(algo_to_run, n_components, dataset, 'binary' if reduce_kdd_to_binary else '')

num_test_runs = 1
train_scores = []
test_scores = []
train_stats_avg = np.zeros((num_unique_classes, 6))
test_stats_avg = np.zeros((num_unique_classes, 6))

train_scores_PCA = []
test_scores_PCA = []
train_stats_avg_PCA = np.zeros((num_unique_classes, 6))
test_stats_avg_PCA = np.zeros((num_unique_classes, 6))

train_scores_ICA = []
test_scores_ICA = []
train_stats_avg_ICA = np.zeros((num_unique_classes, 6))
test_stats_avg_ICA = np.zeros((num_unique_classes, 6))

train_scores_RCA = []
test_scores_RCA = []
train_stats_avg_RCA = np.zeros((num_unique_classes, 6))
test_stats_avg_RCA = np.zeros((num_unique_classes, 6))

train_scores_LDA = []
test_scores_LDA = []
train_stats_avg_LDA = np.zeros((num_unique_classes, 6))
test_stats_avg_LDA = np.zeros((num_unique_classes, 6))
#this is just a random comment to verify this code was written by Boyko Todorov

for test_run_index in range(num_test_runs):
    x_train, x_test, y_train, y_test = data_service. \
        load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                            random_seed=random_seed, dataset=dataset, test_size=test_size)

    if dataset == 'kdd' and reduce_kdd_to_binary:
        y_train, y_test = convert_to_binary_util.convert(y_train.copy(), y_test.copy(), 11)

        # smurf_indices = np.where(y_train == 18)[0]
        # smurfs = x_train[smurf_indices, :]
        # print("Smurfs stats: ")
        # print_attribute_stats(smurfs)

    train_score, class_train_stats, test_score, class_test_stats = \
        cluster_algo.run(x_train, x_test, y_train, y_test, num_unique_classes)

    train_scores.append(train_score)
    test_scores.append(test_score)

    train_stats_avg = stats_util.update_avg_stats(class_train_stats, train_stats_avg, test_run_index)
    test_stats_avg = stats_util.update_avg_stats(class_test_stats, test_stats_avg, test_run_index)

    # print("Applying Reduction...")
    #
    # train_score_reduced_PCA, class_train_stats_reduced_PCA, test_score_reduced_PCA, class_test_stats_reduced_PCA = \
    #     apply_reduction_and_cluster_reduced_data('PCA', n_components, cluster_algo, x_train.copy(), x_test.copy(),
    #                                              y_train, y_test, num_unique_classes, dataset)
    # train_scores_PCA.append(train_score_reduced_PCA)
    # test_scores_PCA.append(test_score_reduced_PCA)
    # train_stats_avg_PCA = stats_util.update_avg_stats(class_train_stats_reduced_PCA, train_stats_avg_PCA, test_run_index)
    # test_stats_avg_PCA = stats_util.update_avg_stats(class_test_stats_reduced_PCA, test_stats_avg_PCA, test_run_index)
    #
    # train_score_reduced_ICA, class_train_stats_reduced_ICA, test_score_reduced_ICA, class_test_stats_reduced_ICA = \
    #     apply_reduction_and_cluster_reduced_data('ICA', n_components, cluster_algo, x_train.copy(), x_test.copy(),
    #                                              y_train, y_test, num_unique_classes, dataset)
    # train_scores_ICA.append(train_score_reduced_ICA)
    # test_scores_ICA.append(test_score_reduced_ICA)
    # train_stats_avg_ICA = stats_util.update_avg_stats(class_train_stats_reduced_ICA, train_stats_avg_ICA, test_run_index)
    # test_stats_avg_ICA = stats_util.update_avg_stats(class_test_stats_reduced_ICA, test_stats_avg_ICA, test_run_index)
    #
    # train_score_reduced_RCA, class_train_stats_reduced_RCA, test_score_reduced_RCA, class_test_stats_reduced_RCA = \
    #     apply_reduction_and_cluster_reduced_data('RCA', n_components, cluster_algo, x_train.copy(), x_test.copy(),
    #                                              y_train, y_test, num_unique_classes, dataset)
    # train_scores_RCA.append(train_score_reduced_RCA)
    # test_scores_RCA.append(test_score_reduced_RCA)
    # train_stats_avg_RCA = stats_util.update_avg_stats(class_train_stats_reduced_RCA, train_stats_avg_RCA, test_run_index)
    # test_stats_avg_RCA = stats_util.update_avg_stats(class_test_stats_reduced_RCA, test_stats_avg_RCA, test_run_index)
    #
    train_score_reduced_LDA, class_train_stats_reduced_LDA, test_score_reduced_LDA, class_test_stats_reduced_LDA = \
        apply_reduction_and_cluster_reduced_data('LDA', n_components, cluster_algo, x_train.copy(), x_test.copy(),
                                                 y_train.astype(np.int64), y_test.astype(np.int64), num_unique_classes, dataset)
    train_scores_LDA.append(train_score_reduced_LDA)
    test_scores_LDA.append(test_score_reduced_LDA)
    train_stats_avg_LDA = stats_util.update_avg_stats(class_train_stats_reduced_LDA, train_stats_avg_LDA, test_run_index)
    test_stats_avg_LDA = stats_util.update_avg_stats(class_test_stats_reduced_LDA, test_stats_avg_LDA, test_run_index)

print("Clustering without Reduction")
print_stats(train_scores, test_scores, train_stats_avg, test_stats_avg, num_test_runs)
# print("Stats after Dimensionality Reduction:")
#
# print("PCA")
# print_stats(train_scores_PCA, test_scores_PCA, train_stats_avg_PCA, test_stats_avg_PCA,
#             num_test_runs)
# print('------------------------------------------------')
#
# print("ICA")
# print_stats(train_scores_ICA, test_scores_ICA, train_stats_avg_ICA, test_stats_avg_ICA,
#             num_test_runs)
# print('------------------------------------------------')
#
# print("RCA")
# print_stats(train_scores_RCA, test_scores_RCA, train_stats_avg_RCA, test_stats_avg_RCA,
#             num_test_runs)
# print('------------------------------------------------')
#
print("LDA")
print_stats(train_scores_LDA, test_scores_LDA, train_stats_avg_LDA, test_stats_avg_LDA,
            num_test_runs)
print('------------------------------------------------')



#
# print("Applying ICA...")
#
# fastICA = FastICA(n_components=3, random_state=0)
# x_train_ICA = fastICA.fit_transform(x_train.copy())
# x_test_ICA = fastICA.transform(x_test.copy())
#
# cluster_algo.run(x_train_ICA, x_test_ICA, y_train, y_test)
#
# print("Applying RCA...")
#
# rca = GaussianRandomProjection(n_components=26)
# x_train_RCA = rca.fit_transform(x_train.copy())
# x_test_RCA = rca.transform(x_test.copy())
#
# cluster_algo.run(x_train_RCA, x_test_RCA, y_train, y_test)
#
# print("Applying LDA...")
#
# rca = LinearDiscriminantAnalysis(n_components=1)
# x_train_LDA = rca.fit_transform(x_train.copy(), y_train)
# x_test_LDA = rca.transform(x_test.copy())
#
# cluster_algo.run(x_train_LDA, x_test_LDA, y_train, y_test)
