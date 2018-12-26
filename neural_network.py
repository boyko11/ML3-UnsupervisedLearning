from sklearn import neural_network
from learner import Learner


class NNLearner(Learner):

    def __init__(self, hidden_layer_sizes=(100, ), max_iter=200, solver='lbfgs', activation='relu', alpha=0.0001,
                 learning_rate='constant', learning_rate_init=0.001):
        #self.estimator = neural_network.MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, max_iter=1)
        self.estimator = neural_network.MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, solver=solver,
            activation=activation, alpha=alpha, random_state=1, max_iter=max_iter, learning_rate=learning_rate,
                                                      learning_rate_init=learning_rate_init)

    def fit_predict_score(self, x_train, y_train, x_test, y_test):

        # mlp_classifier = neural_network.MLPClassifier(hidden_layer_sizes=(64,64))
        return super(NNLearner, self).fit_predict_score(self.estimator, x_train, y_train, x_test, y_test)



