import data_service
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from stats_util import generate_stats
import numpy as np

def run(x_train, x_test, y_train, y_test, n_classes):

    unique_labels, unique_labels_counts = np.unique(y_train, return_counts=True)
    unique_labels_num = unique_labels.shape[0]
    print("Data Has {0} unique labes".format(unique_labels_num))

    #for cov_type in ['spherical', 'diag', 'tied', 'full']:
    cov_type = 'full'
    print("Covariance Type: {0}".format(cov_type))
    em = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=25, random_state=None)
    em.fit(x_train)
    train_prediction = em.predict(x_train)
    test_prediction = em.predict(x_test)

    print("TRAINING:")
    overall_train_accuracy, train_stats_per_class = generate_stats(y_train, train_prediction, n_classes, x_train)
    print("********** END TRAINING ************\n\n")

    print("TESTING:")
    overall_test_accuracy, test_stats_per_class = generate_stats(y_test, test_prediction, n_classes, x_test)
    return overall_train_accuracy, train_stats_per_class, overall_test_accuracy, test_stats_per_class




# scale_data = True
# transform_data = False
# random_slice = None
# random_seed = None
# dataset = 'breast_cancer'
# test_size = 0.2
# n_classes = 2
#
# x_train, x_test, y_train, y_test = data_service.\
#     load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
#                         random_seed=random_seed, dataset=dataset, test_size=test_size)
#
# run(x_train, x_test, y_train, y_test, n_classes)
#
# print('Applying PCA...')
#
# pca = PCA(n_components=10)
# x_train_PCA = pca.fit_transform(x_train.copy())
# x_test_PCA = pca.transform(x_test.copy())
#
# run(x_train_PCA, x_test_PCA, y_train, y_test, n_classes)
#
# print("Applying ICA...")
#
# fastICA = FastICA(n_components=9, random_state=0)
# x_train_ICA = fastICA.fit_transform(x_train.copy())
# x_test_ICA = fastICA.transform(x_test.copy())
#
# run(x_train_ICA, x_test_ICA, y_train, y_test, n_classes)
#
# print("Applying RCA...")
#
# rca = GaussianRandomProjection(n_components=27)
# x_train_RCA = rca.fit_transform(x_train.copy())
# x_test_RCA = rca.transform(x_test.copy())
#
# run(x_train_RCA, x_test_RCA, y_train, y_test)
#
# print("Applying LDA...")
#
# rca = LinearDiscriminantAnalysis(n_components=1)
# x_train_LDA = rca.fit_transform(x_train.copy(), y_train, n_classes)
# x_test_LDA = rca.transform(x_test.copy())
#
# run(x_train_LDA, x_test_LDA, y_train, y_test)



