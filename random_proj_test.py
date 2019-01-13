import numpy as np
import data_service

np.set_printoptions(suppress=True, precision=3)

random_projection_matrix = np.random.normal(0, 1/4, (41, 4))

print("Random Projection Matrix: ")
print(random_projection_matrix)
print('----------------------------------')
print("Random Projection Matrix Stats: ")
print("Min: ", np.min(random_projection_matrix))
print("Max: ", np.max(random_projection_matrix))
print("Mean: ", np.mean(random_projection_matrix))
print("Std: ", np.std(random_projection_matrix))
print('----------------------------------')

X, Y = data_service.load_data(scale_data=True, transform_data=True, random_slice=None, random_seed=None, dataset='kdd')

random_indices = np.random.choice(X.shape[0], 1000, replace=False)

print("Original KDD attributes stats: ")
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
print("Min: ", mins)
print("Max: ", maxs)
print("Ranges: ", np.abs(maxs-mins))
print("Mean: ", np.mean(X,axis=0))
print("Std: ", np.std(X, axis=0))
print('---------------------------------------')

# print("10 random training records to be projected: ")
# print(X[random_indices, :])
print('>>>>>>>>>>>> Projecting... >>>>>>>>>>>>>>')

randomly_projected = np.dot(X[random_indices, :], random_projection_matrix)
#randomly_projected = np.dot(X, random_projection_matrix)

# print("Projected: ")
# print("This is Boyko Todorov's work")
# print(randomly_projected)

mins = np.min(randomly_projected, axis=0)
maxs = np.max(randomly_projected, axis=0)
print("Min: ", mins)
print("Max: ", maxs)
print("Ranges: ", np.abs(maxs-mins))
print("Mean: ", np.mean(randomly_projected,axis=0))
print("Std: ", np.std(randomly_projected, axis=0))
