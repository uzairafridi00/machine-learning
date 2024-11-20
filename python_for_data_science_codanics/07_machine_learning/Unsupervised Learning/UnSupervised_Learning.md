# Unsupervised Machine Learning

A type of machine learning that learns from data without human supervision. Unlike supervised learning, unsupervised machine learning models are given unlabeled data and allowed to discover patterns and insights without any explicit guidance or instruction.

1. No Labels
2. Non Supervisor
3. We have only features there

## Clustering Algorithm

### 1. K Means Cluster

An unsupervised machine learning algorithm that groups data into clusters based on their features. 

- We try to minimize the intracluster variance.
- Out goal is to maximize intercluter variance.

#### How it works

K-means clustering partitions data into k clusters by: 
1. Choosing k initial cluster centers (centroids) 
2. Computing the distance between each observation and each centroid 
3. Assigning each observation to the cluster with the closest centroid 
4. Updating the centroids based on the assigned points 
5. Repeating until the clusters stabilize 

### 2. Hierarichal Clustering

Hierarchical clustering works by creating a cluster tree, or dendrogram, that visually represents the relationships between the data points.

- It contains an order.
- Look at major level similarities > then minor and then again go to minor.

### 3. DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a machine learning algorithm that groups data points into clusters based on their distance to other points.

- Given a set of points in some space, it groups together points that are closely packed (points with many nearby neighbors)
- Marks as outliers points that lie alone in low-density regions (those whose nearest neighbors are too far away).
- A man is know by the company he keeps. (Density)

### 4. Gaussian Mixture Models (GMM)

A soft clustering technique that uses a probability model to group similar data points into clusters.

- GMMs assume that data is a mixture of multiple Gaussian distributions, each representing a distinct cluster.
- The model estimates the parameters of each Gaussian distribution, such as the mean and variance, and the weight of each cluster. It then calculates the probability that each data point belongs to each cluster.

#### When to use GMMs

GMMs are useful when: 

1. Clusters have different shapes and sizes.
2. There is uncertainty in assigning data points 
3. Probabilistic assignments are beneficial 


### 5. Dimensionality Reduction Algorithms

Columns (Features) are data dimensions. We use these for feature selection in order to find target variable.

#### Principle Component Analysis (PCA)

- An unsupervised algorithm that reduces the number of features in large datasets.
- PCA focuses on overall variance and is suitable for unsupervised dimensionality reduction.

#### T-SNE (t-distributed stochastic neighbor embedding)

- T Distribution (Normal)
- It is a statistical method that visualizes high-dimensional data by reducing it to lower dimensions, typically two or three. 
- This technique is useful for exploring and understanding complex datasets, such as those in machine learning and data science. 

#### Autoencoders

- A type of artificial neural network that can be used for dimensionality reduction, or feature selection and extraction.
- They are a deep learning architecture that learns a data representation by copying the input layer to the output layer.

#### SVD (Singular Value Decomposition)

Convert data (Matrix) into single vector.

A linear algebra concept that factorizes a matrix into three special matrices:
- Rotation: The first step in the factorization
- Rescaling: The second step in the factorization
- Another rotation: The final step in the factorization 

### 6. Anamoly Detection Algorithms

A statistical technique that identifies data points that are rare or unusual, and don't conform to a normal pattern.
It's also known as outlier detection.

#### Isolation Forest

Uses an ensemble of Isolation Trees for the given data points to isolate anomalies.

#### Local Outlier Factor (LOF)

- An unsupervised machine learning algorithm that identifies outliers by comparing the density of data points in their local neighborhoods.
- LOF is used to find anomalous data points by measuring how much a given data point deviates from its neighbors.
- A point with a much lower density than its neighbors will have a high LOF score and can be considered an outlier. 

#### One Class SVM

A variant of the traditional SVM algorithm primarily employed for outlier and novelty detection tasks.

### 7. Association Rule Learning Algorithm

### 8. Topic Modelling

### 9. Neural Based Models
