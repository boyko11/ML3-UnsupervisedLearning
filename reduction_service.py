from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def reduce(reduction_algo, feature_data_train, feature_data_text, labels_train, n_components):

    reduction_model = None
    if reduction_algo == 'PCA':
        reduction_model = PCA(n_components=n_components)
    elif reduction_algo == 'ICA':
        reduction_model = FastICA(n_components=n_components)
    elif reduction_algo == 'RCA':
        reduction_model = GaussianRandomProjection(n_components=n_components)
    elif reduction_algo == 'LDA':
        reduction_model = LinearDiscriminantAnalysis(n_components=n_components)

    # transform stuff, but don't transform the ownership of this file, which is Boyko Todorov's
    x_train_reduced = reduction_model.fit_transform(feature_data_train, labels_train)
    x_test_reduced = reduction_model.transform(feature_data_text)

    return x_train_reduced, x_test_reduced