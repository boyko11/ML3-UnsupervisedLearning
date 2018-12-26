from sklearn.metrics import accuracy_score
import time
import numpy as np


class Learner:

    def fit_predict_score(self, learner, x_train, y_train, x_test, y_test):

        start_learning_time = time.time()
        learned_model = learner.fit(x_train, y_train)
        # print("Printing Model: ")
        # print(dir(learned_model))
        # print("End Model")
        learning_time = time.time() - start_learning_time
        # print('learning_time: ', learning_time)
        overall_accuracy_score, predict_time = self.predict_score(learned_model, x_train, y_train, x_test, y_test)
        return overall_accuracy_score, learning_time, predict_time


    def predict_score(self, learned_model, x_train, y_train, x_test, y_test):

        start_prediction_time = time.time()
        prediction = learned_model.predict(x_test)
        predict_time = time.time() - start_prediction_time

        overall_accuracy_score = accuracy_score(y_test, prediction)
        #
        # print('prediction: ', prediction[:20] )
        # print('actual    : ', y_test[:20])
        # print('overall_accuracy_score, prediction_time: ', overall_accuracy_score, predict_time)
        # print('-----------------------------------------------')

        distinct_test_classes = np.unique(y_test)
        print("Class, Accuracy, Train Instances, Test Instances:");
        for class_label in np.nditer(distinct_test_classes):
            class_indices = np.where(y_test == class_label)
            predictions_for_this_class = learned_model.predict(x_test[class_indices])
            number_training_instances = x_train[np.where(y_train == class_label)].shape[0]
            number_testing_instances = predictions_for_this_class.shape[0]
            print("{0},{1:.2f},{2},{3}".format(class_label,
                                                             accuracy_score(y_test[class_indices], predictions_for_this_class), number_training_instances, number_testing_instances))

        print('-----------------------------------------------')

        return overall_accuracy_score, predict_time