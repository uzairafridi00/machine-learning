# Unsupervised Machine Learning

A type of machine learning that learns from data without human supervision. Unlike supervised learning, unsupervised machine learning models are given unlabeled data and allowed to discover patterns and insights without any explicit guidance or instruction.

1. No Labels
2. Non Supervisor
3. We have only features there

## Clustering Algorithm

### 1. K Means Cluster

An unsupervised machine learning algorithm that groups data into clusters based on their features. 

- We try to minimize the intracluster variance.
- Out goal is to increase intercluter variance.

#### How it works

1. K-means clustering partitions data into k clusters by: 
2. Choosing k initial cluster centers (centroids) 
3. Computing the distance between each observation and each centroid 
4. Assigning each observation to the cluster with the closest centroid 
5. Updating the centroids based on the assigned points 
6. Repeating until the clusters stabilize 

### 2. Hierarichal Clustering

Hierarchical clustering works by creating a cluster tree, or dendrogram, that visually represents the relationships between the data points.

- It contains an order.
- Look at major level similarities > then minor and then again go to minor.

### DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a machine learning algorithm that groups data points into clusters based on their distance to other points.

- Given a set of points in some space, it groups together points that are closely packed (points with many nearby neighbors)
- Marks as outliers points that lie alone in low-density regions (those whose nearest neighbors are too far away).
- A man is know by the company he keeps. (Density)

### Gaussian Mixture Models (GMM)

A soft clustering technique that uses a probability model to group similar data points into clusters.

- GMMs assume that data is a mixture of multiple Gaussian distributions, each representing a distinct cluster.
- The model estimates the parameters of each Gaussian distribution, such as the mean and variance, and the weight of each cluster. It then calculates the probability that each data point belongs to each cluster.

#### When to use GMMs

GMMs are useful when: 

1. Clusters have different shapes and sizes.
2. There is uncertainty in assigning data points 
3. Probabilistic assignments are beneficial 


