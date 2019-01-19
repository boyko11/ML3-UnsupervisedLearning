import numpy as np
from neural_network import NNLearner
from service import convert_to_binary_service, reduction_service, data_service
import time
import sys

dataset = 'breast_cancer'
if len(sys.argv) > 1:
    dataset = sys.argv[1]


scale_data = True
transform_data = False
random_slice = None
random_seed = None
test_size = 0.5

nn_activation = 'relu'
alpha = 0.0001
nn_hidden_layer_sizes = (100,)
nn_learning_rate = 'constant'
nn_learning_rate_init = 0.01
nn_solver = 'lbfgs'

num_attributes = 30
num_components = 5

if dataset == 'kdd':
    scale_data = True
    transform_data = True
    random_slice = 2000
    random_seed = None
    test_size = 0.5

    nn_activation = 'relu'
    alpha = 0.01
    nn_hidden_layer_sizes = (100,)
    nn_learning_rate = 'constant'
    nn_learning_rate_init = 0.0001
    nn_solver = 'lbfgs'
    num_attributes = 41
    num_components = 13

num_iter = 100

original_accuracies = []
original_non_reduced_fit_times = []
original_non_reduced_predict_times = []
reduction_algos = ['PCA', 'RCA']

reduction_durations = {'PCA':  [], "RCA": []}
reduced_accuracies = {'PCA':  [], "RCA": []}
reduced_fit_times = {'PCA':  [], "RCA": []}
reduced_predict_times = {'PCA':  [], "RCA": []}

# transform stuff, but don't transform the ownership of this file, which is Boyko Todorov's

for a in range(num_iter):

    x_train, x_test, y_train, y_test = data_service. \
        load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                            random_seed=random_seed, dataset=dataset, test_size=test_size)

    if dataset == 'kdd':
        y_train, y_test = convert_to_binary_service.convert(y_train, y_test, 11)

    nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver,
                           activation=nn_activation,
                           alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

    nn_accuracy_score, nn_fit_time, nn_predict_time = nn_learner.fit_predict_score(x_train.copy(), y_train.copy(),
                                                                                   x_test.copy(), y_test.copy())
    original_accuracies.append(nn_accuracy_score)
    original_non_reduced_fit_times.append(nn_fit_time)
    original_non_reduced_predict_times.append(nn_predict_time)

    print("Iter {0}. Orig score: {1}, fit_time: {2}, predict_time: {3}".format(a, nn_accuracy_score, nn_fit_time, nn_predict_time))

    for reduction_algo in reduction_algos:

        nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver,
                               activation=nn_activation,
                               alpha=alpha, learning_rate=nn_learning_rate,
                               learning_rate_init=nn_learning_rate_init)

        x_train_to_use, x_test_to_use, y_train_to_use, y_test_to_use  = x_train.copy(), x_test.copy(), \
                                                                        y_train.copy(), y_test.copy()

        start_reduction_time = time.time()
        x_train_reduced, x_test_reduced = reduction_service.reduce_train_test_split(reduction_algo, x_train_to_use, x_test_to_use,
                                                                                    y_train_to_use, num_components)
        # x_reduced = np.vstack((x_train_reduced, x_test_reduced))
        # x_reduced_standardized = X = preprocessing.scale(x_reduced)
        # x_train_reduced = x_reduced_standardized[:x_train_reduced.shape[0], :]
        # x_test_reduced = x_reduced_standardized[x_train_reduced.shape[0]:, :]
        reduction_duration = time.time() - start_reduction_time
        reduction_durations[reduction_algo].append(reduction_duration)

        nn_accuracy_score_reduction_algo, nn_fit_time_reduced, nn_predict_time_reduced = \
            nn_learner.fit_predict_score(x_train_reduced, y_train_to_use, x_test_reduced, y_test_to_use)

        reduced_accuracies[reduction_algo].append(nn_accuracy_score_reduction_algo)
        reduced_fit_times[reduction_algo].append(nn_fit_time_reduced)
        reduced_predict_times[reduction_algo].append(nn_predict_time_reduced)

        print("Iter {0}. {1}  score: {2}, fit_time: {3}, predict_time: {4}".format(a, reduction_algo, nn_accuracy_score_reduction_algo,
                                                                                   nn_fit_time_reduced,  nn_predict_time_reduced))

    print("-----------------------------------------------")

mean_orig_score = np.mean(np.asarray(original_accuracies))
mean_RCA_score = np.mean(np.asarray(reduced_accuracies['RCA']))
mean_PCA_score = np.mean(np.asarray(reduced_accuracies['PCA']))

mean_orig_fit_time = np.mean(np.asarray(original_non_reduced_fit_times))
mean_RCA_fit_time = np.mean(np.asarray(reduced_fit_times['RCA']))
mean_PCA_fit_time = np.mean(np.asarray(reduced_fit_times['PCA']))

mean_orig_predict_time = np.mean(np.asarray(original_non_reduced_predict_times))
mean_RCA_predict_time = np.mean(np.asarray(reduced_predict_times['RCA']))
mean_PCA_predict_time = np.mean(np.asarray(reduced_predict_times['PCA']))

mean_RCA_reduction_time = np.mean(np.asarray(reduction_durations['RCA']))
mean_PCA_reduction_time = np.mean(np.asarray(reduction_durations['PCA']))

print("Orig stats: score: {0}, fit_time: {1}, predict_time: {2}".format(mean_orig_score, mean_orig_fit_time, mean_orig_predict_time))
print("RCA  stats: score: {0}, fit_time: {1}, predict_time: {2}, reduction_time: {3}, fit+reduction time: {4}"
      .format(mean_RCA_score, mean_RCA_fit_time, mean_RCA_predict_time, mean_RCA_reduction_time, (mean_RCA_fit_time + mean_RCA_reduction_time)))
print("PCA  stats: score: {0}, fit_time: {1}, predict_time: {2}, reduction_time: {3}, fit+reduction time: {4}"
      .format(mean_PCA_score, mean_PCA_fit_time, mean_PCA_predict_time, mean_PCA_reduction_time, (mean_PCA_fit_time + mean_PCA_reduction_time)))
