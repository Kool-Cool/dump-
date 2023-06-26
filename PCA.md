# Principal Component Analysis (PCA)

Principal component analysis (PCA) is a technique for dimensionality reduction and data visualization. It can be used to find the most important features or components of a dataset, and to project the data onto a lower-dimensional space.

## Mathematics of PCA

The main idea of PCA is to find a linear transformation that maps the original data to a new coordinate system, where the variance of the data is maximized along each axis. The new axes are called principal components (PCs), and they are orthogonal to each other.

The first PC is the direction of maximum variance in the data, and it captures the most information about the data. The second PC is the direction of maximum variance in the data that is orthogonal to the first PC, and so on. The PCs can be computed by finding the eigenvectors and eigenvalues of the covariance matrix of the data.

The covariance matrix of a dataset X with n samples and d features is given by:

$$\Sigma = \frac{1}{n} X^T X$$

The eigenvectors of $\Sigma$ are the PCs, and the eigenvalues are proportional to the amount of variance explained by each PC. The PCs are ordered by decreasing eigenvalues, so that the first PC explains the most variance, and the last PC explains the least variance.

To project the data onto a lower-dimensional space, we can select k PCs that explain most of the variance, and multiply them by the original data. This gives us a new dataset Z with n samples and k features:

$$Z = X W_k$$

where $W_k$ is a matrix containing the k eigenvectors as columns.

## Implementing PCA

To implement PCA in Python, we can use the numpy library for linear algebra operations, and matplotlib for plotting. Here is an example of how to apply PCA to a synthetic dataset with two features:
```
python
# Import numpy library
import numpy as np

# Define a sample dataset with 4 variables and 10 observations
X = np.array([[1, 2, 3, 4],
[2, 3, 4, 5],
[3, 4, 5, 6],
[4, 5, 6, 7],
[5, 6, 7, 8],
[6, 7, 8, 9],
[7, 8, 9, 10],
[8, 9, 10, 11],
[9, 10, 11, 12],
[10, 11, 12, 13]])

# Center the data by subtracting the mean
X_mean = X.mean(axis=0)
X_centered = X - X_mean

# Compute the covariance matrix
cov_matrix = np.cov(X_centered.T)

# Compute the eigenvalues and eigenvectors
eig_values, eig_vectors = np.linalg.eig(cov_matrix)

# Sort the eigenvalues in descending order
eig_pairs = [(eig_values[i], eig_vectors[:, i]) for i in range(len(eig_values))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Choose the top k eigenvalues and eigenvectors
k = 2 # Number of principal components
eig_values_k = [eig_pairs[i][0] for i in range(k)]
eig_vectors_k = [eig_pairs[i][1] for i in range(k)]

# Transform the data into the new coordinate system
eig_vectors_k = np.array(eig_vectors_k).T
X_pca = X_centered.dot(eig_vectors_k)

# Print the results
print("Eigenvalues:")
print(eig_values_k)
print("Eigenvectors:")
print(eig_vectors_k)
print("Transformed data:")
print(X_pca)
```
