import numpy as np
from service import plotting_service, reduction_service, data_service
import sys

#based on https://stackoverflow.com/a/36567821/2948202. Thank you Eikenberg!

if len(sys.argv) < 3:
    print("Usage: python projection_error.py <reduction_algorithm: PCA or RCA> <dataset: kdd or breast_cancer")
    exit()

algo = sys.argv[1]
dataset = sys.argv[2]
n_attributes = 30
transform_data = False
random_slice = None

if dataset == 'kdd':
    n_attributes = 41
    transform_data = True
    random_slice = None

x_original, labels = data_service.load_data(scale_data=True, transform_data=transform_data, random_slice=random_slice,
                                            random_seed=None, dataset=dataset)

projection_losses = []
std_list = []
for n_components in range(1, n_attributes + 1):
    print(n_components)
    trial_mse_list = []
    num_trials = 10 if algo == 'RCA' else 1
    for this_trial in range(num_trials):
        reduction_model = reduction_service.build_reduction_model(algo, n_components)
        x_train_reduced = reduction_model.fit_transform(x_original, labels)
        #x_train_inverse_transformed = reduction_model.inverse_transform(x_train_reduced)
        x_train_inverse_transformed = x_train_reduced.dot(reduction_model.components_) + np.mean(x_original, axis=0)
        mse = np.mean(np.square(x_original - x_train_inverse_transformed))
        trial_mse_list.append(mse)

    projection_losses.append(np.mean(np.asarray(trial_mse_list, np.float64)))
    std_list.append(np.std(np.asarray(trial_mse_list, np.float64)))

print(projection_losses)
plotting_service.plot_projection_losses(projection_losses, dataset, algo)
plotting_service.plot_std_per_component(std_list, dataset, algo)


