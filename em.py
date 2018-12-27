import numpy as np
import data_service
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection

def run_em(x_train, x_test, y_train, y_test):

    #for cov_type in ['spherical', 'diag', 'tied', 'full']:
    cov_type = 'full'
    print("Covariance Type: {0}".format(cov_type))
    em = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=25, random_state=None)
    em.fit(x_train)
    train_prediction = em.predict(x_train)
    test_prediction = em.predict(x_test)
    train_accuracy_score_1 = accuracy_score(y_train, train_prediction)
    train_accuracy_score_2 = accuracy_score(y_train, np.logical_not(train_prediction))
    if train_accuracy_score_1 > train_accuracy_score_2:
        print("Train accuracy: {0}".format(train_accuracy_score_1))
    else:
        print("Train accuracy: {0}".format(train_accuracy_score_2))
        test_prediction = np.logical_not(test_prediction)

    test_accuracy = accuracy_score(y_test, test_prediction)
    print("Test accuracy: {0}".format(test_accuracy))
    print("---------------------")



scale_data = True
transform_data = False
random_slice = None
random_seed = None
dataset = 'breast_cancer'
test_size = 0.4
n_classes = 2

x_train, x_test, y_train, y_test = data_service.\
    load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                        random_seed=random_seed, dataset=dataset, test_size=test_size)

run_em(x_train, x_test, y_train, y_test)

print('Applying PCA...')

pca = PCA(n_components=10)
x_train_PCA = pca.fit_transform(x_train.copy())
x_test_PCA = pca.transform(x_test.copy())

run_em(x_train_PCA, x_test_PCA, y_train, y_test)

print("Applying ICA...")

fastICA = FastICA(n_components=9, random_state=0)
x_train_ICA = fastICA.fit_transform(x_train.copy())
x_test_ICA = fastICA.transform(x_test.copy())

run_em(x_train_ICA, x_test_ICA, y_train, y_test)

print("Applying RCA...")

rca = GaussianRandomProjection(n_components=27)
x_train_RCA = rca.fit_transform(x_train.copy())
x_test_RCA = rca.transform(x_test.copy())

run_em(x_train_RCA, x_test_RCA, y_train, y_test)



