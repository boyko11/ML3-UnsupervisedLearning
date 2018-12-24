from sklearn.cluster import KMeans
import numpy as np
import data_service
from sklearn.metrics import accuracy_score


scale_data = True
transform_data = False
random_slice = None
random_seed = None
dataset = 'breast_cancer'
test_size = 0.0

x_train, x_test, y_train, y_test = data_service.\
    load_and_split_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice,
                        random_seed=random_seed, dataset=dataset, test_size=test_size)

# x_train, y_train = data_service.load_data(scale_data, transform_data, random_slice, random_seed, dataset)

kmeans = KMeans(n_clusters=2, random_state=None).fit(x_train)

train_accuracy_score_1 = accuracy_score(y_train, kmeans.labels_)
train_accuracy_score_2 = accuracy_score(y_train, np.logical_not(kmeans.labels_))

print("Train accuracy 1: {0}".format(train_accuracy_score_1))
print("Train accuracy 2: {0}".format(train_accuracy_score_2))

test_prediction = kmeans.predict(x_test)
if train_accuracy_score_2 > train_accuracy_score_1:
    test_prediction = np.logical_not(test_prediction)


test_accuracy = accuracy_score(y_test, test_prediction)
print("Test Accuracy: {0}".format(test_accuracy))