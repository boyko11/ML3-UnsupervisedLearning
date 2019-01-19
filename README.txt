code: https://github.com/boyko11/ML3-UnsupervisedLearning

Installation assumptions:
python 3+, numpy, sklearn, matplotlib, tabulate, scipy

Run Instructions:

1. python cluster.py kmeans 4 breast_cancer

    a. will run kmeans on the breast_cancer dataset
    b. will produce tabular stats for the clustered data
    c. will reduce the feature data to 4 components using PCA, ICA, RCA
    d. will reduce the feature data to one component using LDA
    e. will run the kmeans for each set of reduced data
    f. will produce tabular stats for the clustered REDUCED data
The '4' could be changed to a desired number of components to reduce the dataset to
-----------------------------------------------------------------------------

2. python cluster.py em 4 breast_cancer

Same as 1., but instead of using kmeans for clustering, it uses ExpectationMaximization
This will also produce the 10 softest scores for each Cluster
-----------------------------------------------------------------------------

3. python cluster.py kmeans 4 kdd

Same as 1., but instead of using the breast_cancer dataset, it uses KDD
-----------------------------------------------------------------------------

4. python cluster.py em 4 kdd

Same as 1., but instead of using kmeans for clustering, it uses ExpectationMaximization.
Also instead of using the breast_cancer dataset, it uses kdd
This will also produce the 10 softest scores for each Cluster
-----------------------------------------------------------------------------

5. python cluster.py em 4 kdd binary

Same as 4., but it will convert the kdd dataset to a binary dataset:
Class 0 will be all the records corresponding to Class 'Normal' in the original datatset
Class 1 will be all the Non-'Normal' records
This will also produce the 10 softest scores for each Cluster
-----------------------------------------------------------------------------

6. python eigenvalues.py

Will Run PCA on both breast_cancer and KDD
Will produce Eignevalues.png in the working directory -
a barchart for the eigenvalues for each component for both reduced datatsets.
Principal Component Number(Rank) - x axis, eigenvalue - y axis.

-----------------------------------------------------------------------------

7. python reduce.py PCA breast_cancer 3

    Will reduce the breast_cancer dataset to 3 components using PCA
    Will produce a 1D, 2D and 3D scatter plots with the Reduced Components for axes:
    In the working dir:
    Reduce1D-PCA-breast_cancer.png
    Reduce2D-PCA-breast_cancer.png
    Reduce1D-PCA-breast_cancer.png

-----------------------------------------------------------------------------

8. python reduce.py PCA kdd 3 - same as 7 but for the KDD dataset - created filenames will reflect the dataset name

-----------------------------------------------------------------------------

9. python reduce.py ICA breast_cancer 3
   python reduce.py ICA kdd 3
   python reduce.py RCA breast_cancer 3
   python reduce.py RCA kdd 3
   python reduce.py LDA breast_cancer 3
   python reduce.py LDA kdd 3

   same as 7 and 8, but using the respective reduction algorithm - the scatter plot files created in the working dir
   will reflect the algorithm and dataset used.
   LDA will only create a 1D plot, since it only projects to one dimension

   for the kdd dataset 'binary' may also be passed as the last argument
   e.g: python reduce.py RCA kdd 3 binary
   note: for LDA binary datasets will produce only 1D plots

-----------------------------------------------------------------------------

10. python projection_error.py PCA breast_cancer
    python projection_error.py PCA kdd
    python projection_error.py RCA breast_cancer
    python projection_error.py RCA kdd

    This commands will produce the ProjectionLoss vs Number of Components line plots in the work directory:
    e.g. ProjectionLoss-RCA-kdd.png, ProjectionSTD-RCA-kdd.png

-----------------------------------------------------------------------------

11. python random_proj_test.py

    This will print out stats(Min, Max, Mean, STD) for the KDD feature data
    Then it will project the feature data to 4 dimensions using RCA
    Then it will print out the same stats for the reduced data


-----------------------------------------------------------------------------

12. python kurtosis.py

    Will produce Number of Components vs Mean Kurtosis line plots in the working dir:
    Kurtosis-breast_cancer.png, Kurtosis-kdd.png

-----------------------------------------------------------------------------

13. python reduced_x_neural_network.py KDD True
    python reduced_x_neural_network.py breast_cancer True

    This reproduces the results from Section "Neural Network on Dimensionally Reduced data"
    Line plot in the working directory, e.g
    NeuralNetReduceAndCluster-breast_cancer.png, NeuralNetReduceAndCluster-kdd.png


-----------------------------------------------------------------------------

14. python nn_time_compare.py

    print comparison score, fit time and predict time for the Neural Network trained with original data vs
    the same stats for the same network trained with RCA reduced data
    the same stats for the same network trained with PCA reduced data
    it also prints the time it take to reduce the original data














