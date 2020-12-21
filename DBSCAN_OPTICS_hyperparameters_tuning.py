"""
Understanding DBSCAN (finding the optimal value for epsilon and # of neighbour point
go to:
https://medium.com/@mohantysandip/a-step-by-step-approach-to-solve-dbscan-algorithms-by-tuning-its-hyper-parameters-93e693a91289
"""
from sklearn.cluster import KMeans
from sklearn import preprocessing #we will normalize our data
import pdb
import pandas as pd
import numpy as np
import re
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as sp
from sklearn.model_selection import train_test_split
import seaborn as sns
from DisSimilarity import dissimilarities
from sklearn.metrics import silhouette_score, silhouette_samples #find the optimal k using sihouette score
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS


url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)
df_sub=df[['latitude','longitude']].copy()


################################################################
# first step
#          How to find ideal no. of clusters for the data
# We can use Silhouette score and kmean to find the optimal number of clusters
#################################################################

"""
finding optimal k using silhouette score
"""
the optimal value for k is 15, and this is also when I use the elbow method
k_range=w=list(range(4,20)) #specify the range of k
for k in k_range:
    clusterer= KMeans(n_clusters=k, init='k-means++', random_state=20)
    cluster_labels=clusterer.fit_predict(df_sub.values)#Compute cluster centers and predict cluster index for each sample
    #the silhouette score gives the av. value for all the samples
    #this gives a percepective  into density and seperation of the formed cluster
    silhouette_avg=silhouette_score(df_sub.values,cluster_labels )
    print(f"for k cluster={k}, the av silhouette_score is {silhouette_avg}") #we can see that k>= 13, we get good score


################################################################
# first two
#     find the optimal epsilon and number of neighboor points
#################################################################
#sort the data and find the distance among its neighbors
# to find the minimum distance between them and plot the minimum distance.
# This will essentially give us the elbow curve to find density of the data points,
# and their minimum distance(eps) values

#1. sort by x, y
df_sub2=df_sub.sort_values(by=['latitude','longitude'])

#find the distance between a point i and its neighboor i+1,
# this will be the minimum distance between a point and its neighboor, since we sort at the beginning
#create a df to store the minimum distance and its index
df2=pd.DataFrame(columns=['index', 'distance'])
for i in range(0, len(df_sub2)-1):
    dist=np.linalg.norm(df_sub2.iloc[i]-df_sub2.iloc[i+1]) #find the Euclidean distance
    df2=df2.append({'index': str(i), 'distance': dist}, ignore_index=True)

#sort to find the min distances
df2=df2.sort_values(by=['distance'])
#plot the distance in the y-axis and its corresponding index
#print(df2.head())
plt.figure(1)
# ax1=plt.subplot2grid((2,1), (0,1))
plt.scatter(df2['index'], df2['distance'])
plt.title("Elbow graph by \n Sorted min distance between a point i and its neighour i+1")
plt.xlabel("index")
#consider the distances <=0.5, filter
filter_distances=df2['distance']<=0.5
df3=df2[filter_distances].sort_values(by=['distance'])
plt.ylabel("distance")
plt.xticks(np.arange(0, len(df2['index']), step=100), rotation=45)
plt.grid(True)
# ax2=plt.subplot2grid((2,1), (1,1))
plt.figure(2)
plt.scatter(df3['index'], df3['distance'])
plt.title("Elbow graph by \n Sorted min distance (<=0.5) between a point i and its neighour i+1")
plt.xlabel("index")
plt.ylabel("distance")
plt.xticks(np.arange(0, len(df3['index']), step=100), rotation=45)
plt.grid(True)
plt.show()

# from the above graph, we can see that the elbow, is when the distance=0.2


################################################################
# first three
#    Silhouette distance to find ideal eps value for DBSCAN
#################################################################
#although from the above graph, we can see that with epsilon=0.2 we can find the minimum distances,
# but we are going to use min distances between 0.1 and 0.5 and find the silhouette score using DBSCAN,
# then we find the best epsilon the one that has the maximum Silhouette distance
# Note: Here we have assumed the ‘min_samples’ parameter to be 15 which can be changed later.
#15 is the number of clusters that we find it using K-mean
range_eps=[0.1,0.2,0.25,0.26,0.27,0.28,0.29, 0.3,0.35, 0.4, 0.5, 0.6,0.7,0.8,0.9]
for i in range_eps:
    db=DBSCAN(eps=i, min_samples=15).fit(df_sub.values)
    labels=db.labels_
    silhouette_avg=silhouette_score(df_sub.values, labels)
    print(f"for epsilon {i}, the silhouette score is {silhouette_score(df_sub.values, labels):0.3f}")
# for epsilon 0.1, the silhouette score is -0.575
# for epsilon 0.2, the silhouette score is 0.169
# for epsilon 0.25, the silhouette score is 0.271 #I will use this
# for epsilon 0.26, the silhouette score is 0.277
# for epsilon 0.27, the silhouette score is 0.285
# for epsilon 0.28, the silhouette score is 0.294
# for epsilon 0.29, the silhouette score is 0.303
# for epsilon 0.3, the silhouette score is 0.320
# for epsilon 0.35, the silhouette score is 0.347
# for epsilon 0.4, the silhouette score is 0.374
# for epsilon 0.5, the silhouette score is 0.449
# for epsilon 0.6, the silhouette score is 0.464
# for epsilon 0.7, the silhouette score is 0.371
# for epsilon 0.8, the silhouette score is 0.434
# for epsilon 0.9, the silhouette score is 0.485


################################################################
# first four
#    Silhouette distance to find ideal number of neighboor points
#################################################################
#after finding the minimum epsilon, find the number of neighboor points
range_sample=list(range(5,50))
for i in range_sample:
    db=DBSCAN(eps=0.25, min_samples=i).fit(df_sub.values)
    labels=set([label for label in db.labels_ if label>=0]) #ignore the -1 label (noise, outlier label)
    print(f"for min neighboor points {i}, the number of cluster is {labels}, the min number of cluster{len(labels)}")

#we extract the minimum number of clusters
# #we should use
# for min neighboor points=31, to get
# the number of cluster is {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, the min number of cluster15



################################################################
# first five
#   Use Optics Algorithm to prove the ‘min_samples’ value chosen is correct
#################################################################
# Ordering points to identify the clustering structure (OPTICS)
# is an algorithm for finding density-based clusters in spatial data.
# Additionally, a special distance is stored for each point that represents
# the density that must be accepted for a cluster
# so that both points belong to the same cluster.
# It use core distance and reachability distance method to identify the clusters.
# It has only one hyper parameter i.e ‘min_samples’ used to find the clusters.
# Internally it uses DBSCAN method.

#I will use the same for DBSCAN
clustering = OPTICS(min_samples=31).fit(df_sub.values)