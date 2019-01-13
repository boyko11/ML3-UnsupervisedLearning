import numpy as np


def convert(y_train, y_test, class_to_convert_to_zero):

    normal_train_indices = np.where(y_train == class_to_convert_to_zero)[0]
    non_normal_train_indices = np.where(y_train != class_to_convert_to_zero)[0]
    y_train[normal_train_indices] = 0
    y_train[non_normal_train_indices] = 1
    normal_test_indices = np.where(y_test == class_to_convert_to_zero)[0]
    non_normal_test_indices = np.where(y_test != class_to_convert_to_zero)[0]
    y_test[normal_test_indices] = 0
    y_test[non_normal_test_indices] = 1

    return y_train.astype(np.int64), y_test.astype(np.int64)

#util created by Boyko Todorov as part of Project 3 - OMSCS ML - 2019-01-12