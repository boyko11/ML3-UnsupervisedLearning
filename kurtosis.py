import numpy as np
from service import plotting_service, reduction_service, data_service
from scipy.stats import kurtosis


def get_mean_kurtosis_vector(feature_data_original, the_labels, n_attributes):

    mean_kurtosis_per_number_components = []
    for n_components in range(1, n_attributes + 1):
        print(n_components)
        num_trials = 1
        for this_trial in range(num_trials):
            reduction_model = reduction_service.build_reduction_model("ICA", n_components)
            x_train_reduced = reduction_model.fit_transform(feature_data_original, the_labels)
            kurtosis_vector = kurtosis(x_train_reduced, axis=0)
            mean_kurtosis = np.mean(kurtosis_vector)
            print(kurtosis_vector)
            print(mean_kurtosis)
            mean_kurtosis_per_number_components.append(mean_kurtosis)

    return mean_kurtosis_per_number_components


np.set_printoptions(suppress=True, precision=3)

x_original_breast_cancer, labels_breast = data_service.load_data(scale_data=True, transform_data=False, random_slice=None,
                                                                 random_seed=None, dataset='breast_cancer')
print("Breast Cancer original feature data Kurtosis")
kurtosis_vector_bc_orig = kurtosis(x_original_breast_cancer, axis=0)
print(kurtosis_vector_bc_orig)
print("Breast Cancer Mean kurtosis: {0}".format(np.mean(kurtosis_vector_bc_orig)))

x_original_kdd, labels_kdd = data_service.load_data(scale_data=True, transform_data=True, random_slice=2000,
                                                    random_seed=None, dataset='kdd')
print("KDD original feature data Kurtosis")
kurtosis_vector_kdd_orig = kurtosis(x_original_kdd, axis=0)
print(kurtosis_vector_kdd_orig)
print("KDD Mean kurtosis: {0}".format(np.mean(kurtosis_vector_kdd_orig)))

breast_cancer_mean_kurtosis_vector = get_mean_kurtosis_vector(x_original_breast_cancer, labels_breast, 30)


plotting_service.plot_component_stat_line(breast_cancer_mean_kurtosis_vector, 'Mean Kurtosis', 'Mean Kurtosis Per IC', 'breast_cancer')


kdd_mean_kurtosis_vector = get_mean_kurtosis_vector(x_original_kdd, labels_kdd, 41)

plotting_service.plot_component_stat_line(kdd_mean_kurtosis_vector, 'Mean Kurtosis', 'Mean Kurtosis Per IC', 'kdd')






