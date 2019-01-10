from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
import data_service
from neural_network import NNLearner
import plotting_service
import sys

if len(sys.argv) < 2:
    print("Assuming PCA as data reduction algorithm.")

reduction_algo = sys.argv[1].strip() if len(sys.argv) >= 2 else "PCA"

scale_data = True
transform_data = False
random_slice = None
random_seed = None
dataset = 'breast_cancer'
test_size = 0.4

nn_activation = 'relu'
alpha = 0.0001
nn_hidden_layer_sizes = (100,)
nn_learning_rate = 'constant'
nn_learning_rate_init = 0.01
nn_solver = 'lbfgs'


num_iter = 5
num_attributes = 30

original_non_pca_scores = []
number_of_pcs_to_match_or_better_score = []
iter_pcs_scores = np.zeros((num_iter, num_attributes))
for a in range(num_iter):

    x_train, x_test, y_train, y_test = data_service. \
        load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                            random_seed=random_seed, dataset=dataset, test_size=test_size)

    nn_learner = NNLearner(hidden_layer_sizes=nn_hidden_layer_sizes, max_iter=200, solver=nn_solver, activation=nn_activation,
                           alpha=alpha, learning_rate=nn_learning_rate, learning_rate_init=nn_learning_rate_init)

    nn_accuracy_score, nn_fit_time, nn_predict_time = nn_learner.fit_predict_score(x_train.copy(), y_train.copy(), x_test.copy(), y_test.copy())
    original_non_pca_scores.append(nn_accuracy_score)

    print("Iter {0}. Original score: {1}".format(a, nn_accuracy_score))
    print('-----------------------------------------------')

    matching_or_better_score_found = False
    for i in range(1,x_train.shape[1] + 1):

        reduction_model = None
        if reduction_algo == 'PCA':
            reduction_model = PCA(n_components=i)
        elif reduction_algo == 'ICA':
            reduction_model = FastICA(n_components=i)
        elif reduction_algo == 'RCA':
            reduction_model = GaussianRandomProjection(n_components=i)
        elif reduction_algo == 'LDA':
            reduction_model = LinearDiscriminantAnalysis(n_components=i)

        # transform stuff, but don't transform the ownership of this file, which is Boyko Todorov's
        x_train_transformed = reduction_model.fit_transform(x_train.copy(), y_train)
        x_test_transformed = reduction_model.transform(x_test.copy())

        nn_accuracy_score_pca, nn_fit_time_pca, nn_predict_time_pca = \
            nn_learner.fit_predict_score(x_train_transformed, y_train.copy(), x_test_transformed.copy(), y_test.copy())

        score_diff = nn_accuracy_score_pca - nn_accuracy_score
        fit_time_diff = nn_fit_time_pca - nn_fit_time
        predict_time_diff = nn_predict_time_pca - nn_predict_time
        print('Number of principal components: {0}, score: {1}, score_diff: {2}'.format(i, nn_accuracy_score_pca, score_diff))
        print('-----------------------------------------------')
        if score_diff >= 0.0 and not matching_or_better_score_found:
            number_of_pcs_to_match_or_better_score.append(i)
            matching_or_better_score_found = True

        iter_pcs_scores[a, i - 1] = nn_accuracy_score_pca

mean_scores_per_number_pcs = np.mean(iter_pcs_scores, axis=0)

print(original_non_pca_scores)
print(np.mean(np.asarray(original_non_pca_scores)))

plotting_service.plot_scores_per_pcs(mean_scores_per_number_pcs, np.mean(np.asarray(original_non_pca_scores)), reduction_algo)




