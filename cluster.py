import numpy as np
import data_service
import sys
from tabulate import tabulate
import stats_util
import kmeans
import em


np.set_printoptions(suppress=True, precision=3)
if len(sys.argv) < 3 or (sys.argv[1] != 'kmeans' and sys.argv[1] != 'em') \
        or (sys.argv[2] != 'breast_cancer' and sys.argv[2] != 'kdd'):
    print("Usage: python cluster.py kmeans breast_cancer")
    print("or")
    print("Usage: python cluster.py kmeans kdd")
    print("or")
    print("Usage: python cluster.py kmeans kdd binary")
    print("or")
    print("Usage: python cluster.py em breast_cancer")
    print("or")
    print("Usage: python cluster.py em kdd")
    print("or")
    print("Usage: python cluster.py em kdd binary")
    exit()

cluster_algo = kmeans
algo_to_run = sys.argv[1]
if algo_to_run == 'em':
    cluster_algo = em

dataset = sys.argv[2]

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
    if len(sys.argv) > 3:
        reduce_kdd_to_binary = sys.argv[3] == 'binary'
        if reduce_kdd_to_binary:
            num_unique_classes = 2

num_test_runs = 1
train_scores = []
test_scores = []
train_stats_avg = np.zeros((num_unique_classes, 6))
test_stats_avg = np.zeros((num_unique_classes, 6))
for test_run_index in range(num_test_runs):
    x_train, x_test, y_train, y_test = data_service. \
        load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                            random_seed=random_seed, dataset=dataset, test_size=test_size)

    if dataset == 'kdd' and reduce_kdd_to_binary:
        normal_train_indices = np.where(y_train == 9)[0]
        non_normal_train_indices = np.where(y_train != 9)[0]
        y_train[normal_train_indices] = 0
        y_train[non_normal_train_indices] = 1
        normal_test_indices = np.where(y_test == 9)[0]
        non_normal_test_indices = np.where(y_test != 9)[0]
        y_test[normal_test_indices] = 0
        y_test[non_normal_test_indices] = 1

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

# print("Applying PCA...")
#
# pca = PCA(n_components=10)
# x_train_PCA = pca.fit_transform(x_train.copy())
# x_test_PCA = pca.transform(x_test.copy())
#
# cluster_algo.run(x_train_PCA, x_test_PCA, y_train, y_test)
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
