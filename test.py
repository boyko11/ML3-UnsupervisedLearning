# a = np.random.randint(10, size=10)
# print(a)
#
# print(-a)
# print((-a).argsort())
# print((-a).argsort()[:3])

#print(np.isin([1,2,3,4,5,8,6,1,1],[1, 2]))

# a = np.array([1, 2, 3])
# b = np.array([4, 5, 6])
# zipped = zip(a, b)
# print(zipped)
# dicted = dict(zipped)
# print(dicted)

# print(np.random.choice(10, 8, replace=False))

# a = np.array([[1,4, 10, 0],
#               [3,1, 0, 10]])
#
# a.sort(axis=0)
#
# print(a)
# np.set_printoptions(suppress=True, precision=3)
# random_projection_matrix = np.random.normal(0, 1/4, (41, 4))
# print(random_projection_matrix)
#
# print('----------------------------------')
#
#
# X, Y = data_service.load_data(scale_data=True, transform_data=True, random_slice=None, random_seed=None, dataset='kdd')
# print("Original: ")
# print("Min: ", np.min(X, axis=0))
# print("Max: ", np.max(X, axis=0))
# print("Mean: ", np.mean(X,axis=0))
# print("Std: ", np.std(X, axis=0))
#
# print('---------------------------------------')
#
# random_indices = np.random.choice(Y.shape[0], 10000, replace=False)
# print("Random Projection Matrix: ")
# print("Min: ", np.min(random_projection_matrix))
# print("Max: ", np.max(random_projection_matrix))
# print("Mean: ", np.mean(random_projection_matrix))
# print("Std: ", np.std(random_projection_matrix))
#
#
# print(X[random_indices, :])
# print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
#
# randomly_projected = np.dot(X[random_indices, :], random_projection_matrix)
# #randomly_projected = np.dot(X, random_projection_matrix)
#
# print(randomly_projected)
#
# print("Transformed: ")
# print("Min: ", np.min(randomly_projected, axis=0))
# print("Max: ", np.max(randomly_projected, axis=0))
# print("Mean: ", np.mean(randomly_projected,axis=0))
# print("Std: ", np.std(randomly_projected, axis=0))

for i in range(13, 14):
    print(i)


