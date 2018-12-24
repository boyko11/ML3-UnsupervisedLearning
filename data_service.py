from sklearn import datasets,  model_selection, preprocessing
import numpy as np


def load_and_split_data(scale_data=False, transform_data=False, test_size=0.5, random_slice=None, random_seed=None, dataset='breast_cancer'):

    X, Y = load_data(scale_data=scale_data, transform_data=transform_data, random_slice=random_slice, random_seed=random_seed, dataset=dataset)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size)
    return X_train, X_test, Y_train, Y_test


def load_data(scale_data=False, transform_data=False, random_slice=None, random_seed=None, dataset='breast_cancer'):

    if random_seed is not None:
        np.random.seed(random_seed)

    if dataset == 'breast_cancer':
        data = datasets.load_breast_cancer()
    elif dataset == 'kdd':
        data = datasets.fetch_kddcup99()
    #data = datasets.fetch_covtype()

    X = data.data

    Y = data.target

    #np.savetxt("/home/btodorov/Desktop/foo.csv", X[np.random.choice(Y.shape[0], 1000, replace=False), :], delimiter=",")

    ten_random_records = np.random.choice(Y.shape[0], 10, replace=False)
    # print(X[ten_random_records, :])
    # print('-----------------------------------------------')
    # print(Y[ten_random_records])
    # print('-----------------------------------------------')
    # print('X.shape: ', X.shape)
    # print('Y.shape: ', Y.shape)
    # print('-----------------------------------------------')
    if random_slice is not None:
        random_indices = np.random.choice(Y.shape[0], random_slice if random_slice < Y.shape[0] else Y.shape[0], replace=False)
        X = X[random_indices, :]
        Y = Y[random_indices]

    if transform_data:
        for i in [1, 2, 3]:
            print(X[0, i])
            le = preprocessing.LabelEncoder()
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])
            #print('Min-Max {0}: {1}-{2}'.format(i, np.min(X[:, i]), np.max(X[:, i])))
        le = preprocessing.LabelEncoder()
        le.fit(Y)
        Y = le.transform(Y)

    # print(np.amin(X, axis=0))
    # print(np.amax(X, axis=0))
    # print(np.var(X, axis=0))
    # print('1-----------------------------------------------')
    if scale_data:
        X = preprocessing.scale(X)
        #X = preprocessing.MinMaxScaler().fit_transform(X)
        # for i in range(X.shape[1]):
        #     print('Min-Max {0}: {1}-{2}'.format(i, np.min(X[:, i]), np.max(X[:, i])))

        # print(np.amin(X, axis=0))
        # print(np.amax(X, axis=0))
        # print(np.var(X, axis=0))
        # print('2-----------------------------------------------')

    shuffled_indices = np.random.choice(Y.shape[0], Y.shape[0], replace=False)

    X_shuffled = X[shuffled_indices, :]
    Y_shuffled = Y[shuffled_indices]

    return X_shuffled, Y_shuffled
