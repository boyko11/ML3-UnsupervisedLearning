from sklearn import datasets,  model_selection, preprocessing
import numpy as np

def get_random_slice(dataset_name, random_slice, X, Y):

    if dataset_name != 'kdd':
        random_indices = np.random.choice(Y.shape[0], random_slice if random_slice < Y.shape[0] else Y.shape[0], replace=False)
        X = X[random_indices, :]
        Y = Y[random_indices]
        return X, Y

    Y_rare_classes_indices = np.where(np.isin(Y, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22])))[0]
    Y_dominant_classes_indices = np.where(np.isin(Y, np.array([9, 11, 18])))[0]
    X_rare = X[Y_rare_classes_indices, :]
    Y_rare = Y[Y_rare_classes_indices]

    X_dominant = X[Y_dominant_classes_indices, :]
    Y_dominant = Y[Y_dominant_classes_indices]

    random_indices = np.random.choice(Y_dominant_classes_indices.shape[0], random_slice if random_slice < Y_dominant_classes_indices.shape[0] else Y_dominant_classes_indices.shape[0], replace=False)

    X_dominant_random = X_dominant[random_indices, :]
    Y_dominant_random = Y_dominant[random_indices]

    X = np.vstack((X_rare, X_dominant_random))
    Y = np.hstack((Y_rare, Y_dominant_random))

    return X, Y

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
    elif dataset == 'covtype':
        data = datasets.fetch_covtype()
    #data = datasets.fetch_covtype()

    X = data.data

    Y = data.target

    distinct_labels, labels_record_counts = np.unique(Y, return_counts=True)

    # print("Distinct Labels - {0}:".format(distinct_labels.shape[0]))
    # print(distinct_labels)
    # print("Distinct Labels Record Counts: ")
    # print(labels_record_counts)
    # print('-')
    #makes sure for kdd the labels always map to the same classes
    for label_index in range(distinct_labels.shape[0]):
        indices_for_this_label = np.where(Y == distinct_labels[label_index])
        Y[indices_for_this_label] = label_index

    distinct_labels, labels_record_counts = np.unique(Y, return_counts=True)
    # print(Y.shape[0])
    #
    # print("Distinct Labels - {0}:".format(distinct_labels.shape[0]))
    # print(distinct_labels)
    # print("Distinct Labels Record Counts: ")
    # print(labels_record_counts)
    # print(np.histogram(Y, bins=np.arange(distinct_labels.shape[0]+1))[0])
    # print('--')


    #np.savetxt("/home/btodorov/Desktop/foo.csv", X[np.random.choice(Y.shape[0], 1000, replace=False), :], delimiter=",")

    # ten_random_records = np.random.choice(Y.shape[0], 10, replace=False)
    # print(X[ten_random_records, :])
    # print('-----------------------------------------------')
    # print(Y[ten_random_records])
    # print('-----------------------------------------------')
    # print('X.shape: ', X.shape)
    # print('Y.shape: ', Y.shape)
    # print('-----------------------------------------------')
    if random_slice is not None:
        X_random, Y_random = get_random_slice(dataset, random_slice, X, Y)
        X = X_random
        Y = Y_random

    if transform_data:
        for i in [1, 2, 3]:
            print(X[0, i])
            le = preprocessing.LabelEncoder()
            le.fit(X[:, i])
            X[:, i] = le.transform(X[:, i])
            #print('Min-Max {0}: {1}-{2}'.format(i, np.min(X[:, i]), np.max(X[:, i])))

        #already done to make sure the samle clas always maps to the same numeric value
        # le = preprocessing.LabelEncoder()
        # le.fit(Y)
        # Y = le.transform(Y)

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

    # unique_labels, unique_labels_counts = np.unique(Y, return_counts=True)
    # print("Unique classes labels and their record counts: ")
    # print(tuple(zip(unique_labels, unique_labels_counts)))
    # print("1Data Has {0} unique clusters".format(unique_labels.shape[0]))

    # if unique_labels.shape[0] > 2:
    #     three_most_prevelant_classes_indices = (-unique_labels_counts).argsort()[:3]
    #     #not necessary = unique_labes are already sorted
    #     #three_most_prevelant_classes = unique_labels[three_most_prevelant_classes_indices]
    #     indices_of_records_for_most_common_classes = np.isin(Y, three_most_prevelant_classes_indices)
    #     Y = Y[indices_of_records_for_most_common_classes]
    #     X = X[indices_of_records_for_most_common_classes, :]
    #     print("Most Popular Classes and their Record Counts: ")
    #     print(three_most_prevelant_classes_indices)
    #     print(unique_labels_counts[three_most_prevelant_classes_indices])

    shuffled_indices = np.random.choice(Y.shape[0], Y.shape[0], replace=False)

    X_shuffled = X[shuffled_indices, :]
    Y_shuffled = Y[shuffled_indices]

    return X_shuffled, Y_shuffled
