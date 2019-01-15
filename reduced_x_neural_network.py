import numpy as np
import data_service
from neural_network import NNLearner
import plotting_service
import sys
import convert_to_binary_service
import reduction_service
import kmeans

dataset = 'breast_cancer'
cluster = False
if len(sys.argv) > 1:
    dataset = sys.argv[1]

if len(sys.argv) > 2:
    cluster = bool(sys.argv[2])

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

num_iter = 10

original_non_reduced_data_scores = []
number_of_pcs_to_match_or_better_score = []
reduction_algos = ['PCA', 'ICA', 'RCA', 'LDA']
iter_pcs_scores_pca = np.zeros((num_iter, num_attributes))
iter_pcs_scores_ica = np.zeros((num_iter, num_attributes))
iter_pcs_scores_rca = np.zeros((num_iter, num_attributes))
iter_pcs_scores_lda = np.zeros((num_iter, num_attributes))
reduction_algos_iter_scores = {'PCA': iter_pcs_scores_pca, 'ICA': iter_pcs_scores_ica, 'RCA': iter_pcs_scores_rca,
                               'LDA': iter_pcs_scores_lda}

iter_pcs_scores_pca_plus_c = np.zeros((num_iter, num_attributes))
iter_pcs_scores_ica_plus_c = np.zeros((num_iter, num_attributes))
iter_pcs_scores_rca_plus_c = np.zeros((num_iter, num_attributes))
iter_pcs_scores_lda_plus_c = np.zeros((num_iter, num_attributes))
reduction_algos_iter_scores_plus_c = {'PCA': iter_pcs_scores_pca_plus_c, 'ICA': iter_pcs_scores_ica_plus_c, 'RCA': iter_pcs_scores_rca_plus_c,
                               'LDA': iter_pcs_scores_lda_plus_c}

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
    original_non_reduced_data_scores.append(nn_accuracy_score)

    print("Iter {0}. Original score: {1}".format(a, nn_accuracy_score))
    print('-----------------------------------------------')

    matching_or_better_score_found = False
    for i in range(1, num_attributes + 1):
        for reduction_algo_name in reduction_algos:
            nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver,
                                   activation=nn_activation,
                                   alpha=alpha, learning_rate=nn_learning_rate,
                                   learning_rate_init=nn_learning_rate_init)

            x_train_to_use, x_test_to_use, y_train_to_use, y_test_to_use  = x_train.copy(), x_test.copy(), \
                                                                            y_train.copy(), y_test.copy()

            x_train_reduced, x_test_reduced = reduction_service.reduce(reduction_algo_name, x_train_to_use, x_test_to_use,
                                                                y_train_to_use, i)

            nn_accuracy_score_reduction_algo, nn_fit_time_reduced, nn_predict_time_reduced = \
                nn_learner.fit_predict_score(x_train_reduced, y_train_to_use, x_test_reduced, y_test_to_use)
            reduction_algos_iter_scores[reduction_algo_name][a, i - 1] = nn_accuracy_score_reduction_algo

            if cluster:
                print("Clustering")
                train_clusters, test_clusters = kmeans.train_and_test(x_train_reduced.copy(), x_test_reduced.copy(), y_train_to_use, 2)
                x_train_reduced_plus_c = np.append(x_train_reduced, train_clusters.reshape((len(train_clusters), 1)),
                                                axis=1)
                x_test_reduced_plus_c = np.append(x_test_reduced, test_clusters.reshape((len(test_clusters), 1)),
                                               axis=1)
                nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver,
                                       activation=nn_activation,
                                       alpha=alpha, learning_rate=nn_learning_rate,
                                       learning_rate_init=nn_learning_rate_init)
                nn_accuracy_score_plus_c, nn_fit_time_reduced_plus_c, nn_predict_time_reduced_plus_c = \
                    nn_learner.fit_predict_score(x_train_reduced_plus_c, y_train_to_use, x_test_reduced_plus_c, y_test_to_use)
                reduction_algos_iter_scores_plus_c[reduction_algo_name][a, i - 1] = nn_accuracy_score_plus_c


        # score_diff = nn_accuracy_score_reduced - nn_accuracy_score
        #
        # fit_time_diff = nn_fit_time_reduced - nn_fit_time
        # predict_time_diff = nn_predict_time_reduced - nn_predict_time
        # print('Number of principal components: {0}, score: {1}, score_diff: {2}'.format(i, nn_accuracy_score_reduced, score_diff))
        # print('-----------------------------------------------')
        # if score_diff >= 0.0 and not matching_or_better_score_found:
        #     number_of_pcs_to_match_or_better_score.append(i)
        #     matching_or_better_score_found = True

mean_scores_per_number_pcs = np.mean(reduction_algos_iter_scores['PCA'], axis=0)
mean_scores_per_number_ics = np.mean(reduction_algos_iter_scores['ICA'], axis=0)
mean_scores_per_number_rcs = np.mean(reduction_algos_iter_scores['RCA'], axis=0)
mean_scores_per_number_ldacs = np.mean(reduction_algos_iter_scores['LDA'], axis=0)

mean_scores_per_number_pcs_plus_c = np.mean(reduction_algos_iter_scores_plus_c['PCA'], axis=0)
mean_scores_per_number_ics_plus_c = np.mean(reduction_algos_iter_scores_plus_c['ICA'], axis=0)
mean_scores_per_number_rcs_plus_c = np.mean(reduction_algos_iter_scores_plus_c['RCA'], axis=0)
mean_scores_per_number_ldacs_plus_c = np.mean(reduction_algos_iter_scores_plus_c['LDA'], axis=0)

mean_scores_original = np.mean(np.asarray(original_non_reduced_data_scores))

print(original_non_reduced_data_scores)
print(np.mean(np.asarray(original_non_reduced_data_scores)))

plotting_service.plot_scores_per_pcs(mean_scores_per_number_pcs, mean_scores_per_number_ics, mean_scores_per_number_rcs,
                                     mean_scores_per_number_ldacs, mean_scores_per_number_pcs_plus_c, mean_scores_per_number_ics_plus_c, mean_scores_per_number_rcs_plus_c,
                                     mean_scores_per_number_ldacs_plus_c, mean_scores_original, dataset)
