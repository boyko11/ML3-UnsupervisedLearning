import numpy as np
import data_service
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture


scale_data = True
transform_data = False
random_slice = None
random_seed = None
dataset = 'breast_cancer'
test_size = 0.2
n_classes = 2

x_train, x_test, y_train, y_test = data_service.\
    load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                        random_seed=random_seed, dataset=dataset, test_size=test_size)

#x_train, y_train = data_service.load_data(scale_data, transform_data, random_slice, random_seed, dataset)

for cov_type in ['spherical', 'diag', 'tied', 'full']:
    print("Covariance Type: {0}".format(cov_type))
    em = GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=None)
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
