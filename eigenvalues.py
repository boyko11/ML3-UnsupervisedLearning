import numpy as np
from sklearn.decomposition import PCA
import data_service
import plotting_service


def get_PCA_eigenvalues(transform_data, dataset, n_components):
    X, Y = data_service.load_data(scale_data=True, transform_data=transform_data, random_slice=None, random_seed=None,
                                  dataset=dataset)

    pca = PCA(n_components=n_components, whiten=True)
    x_train_reduced = pca.fit_transform(X, Y)

    # Idirer, Obtain eigen values and vectors from sklearn PCA, stackoverflow.com, https://stackoverflow.com/a/31941631/2948202

    n_samples = X.shape[0]
    # We center the data and compute the sample covariance matrix.
    X -= np.mean(X, axis=0)
    cov_matrix = np.dot(X.T, X) / n_samples

    print("Manually calculated eigenvalues")
    for eigenvector in pca.components_:
        print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))

    print('-------------------------------------')

    print("Eignenvalues from sklearn:")
    print(pca.explained_variance_)

    return pca.explained_variance_


eigenvalues_breast_cancer = get_PCA_eigenvalues(False, 'breast_cancer', 30)
eigenvalues_kdd = get_PCA_eigenvalues(True, 'kdd', 41)

plotting_service.plot_eigenvalues(eigenvalues_breast_cancer, eigenvalues_kdd)
plotting_service.plot_eignevalues_barchart(eigenvalues_breast_cancer, eigenvalues_kdd)
#Boyko Todorov
