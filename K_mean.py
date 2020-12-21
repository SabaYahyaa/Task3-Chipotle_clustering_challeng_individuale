"""
This to understand what I will compute, the inertia, dissimilarity, similarity, ....
Also, I want to know how I will visualize my output, when I have only 2 clusters
K mean, k=2
x='latitude', y='longitude',
Note: we do not need to standarize our data since features have the same scales
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


url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)

df_kmean=df[['latitude','longitude']].copy()

# number of pokemon clusters
state_size = 2#'k-means++' is to initiate the centrodic in a smart way, to get faster convergence
kmeans=KMeans(n_clusters=state_size, init='k-means++').fit(df_kmean.values)



# add cluster index to dataframe, this will e your observation
df_kmean['cluster']=kmeans.labels_

#compute the dissimilarities using Euclidean distance and create a new column in the df
all_observation=df_kmean.values
# clusters is an attribute of the object
cluster_centers = kmeans.cluster_centers_
#compute the dissimilarities
diss=dissimilarities(all_observation, cluster_centers, distance_type='Eculidean')

#add the dis-similarity
df_kmean["Dissimilarities"]=diss

#add the inertia
df_kmean["Inertia"]=kmeans.inertia_

"""
Note: we see that the distortion is high while the dissimilarity is low, so we do not have a good value of k
"""

#get the av inertia (distortion of each cluster) of each cluster
av_inertia_per_cluster=df_kmean.groupby('cluster')["Inertia"].mean()
print(f"The av_inertia_per_cluster (the distortion per cluster)\n {av_inertia_per_cluster}")

#get the average dissimilarity for each class
av_dis=df_kmean.groupby('cluster')["Dissimilarities"].mean()
print(f"The av of dissimilarity per cluster\n {av_dis}")




#get the inertia for each cluster, when the similarity
# (interia is less this better because the distance between a point and its centroide small),
# when the distance is small means that the similarity is large between a point and its centroid
min_inertia_per_cluster=df_kmean.loc[df_kmean.groupby('cluster')["Inertia"].idxmin()]
print(f"The minimum distortion\n {min_inertia_per_cluster}")


#get the dissimilarity for each sample, when the dissimilarity is high this is better
# the dissimilarity is the sumation of distances of each sample and all centroids except the centroide of the sample itself
r=df_kmean.loc[df_kmean.groupby('cluster')["Dissimilarities"].idxmax()]
print(f"The minimum distortion\n {r}")

"""
visualization
"""
# plot data with the type color palette
sns.lmplot(x='latitude', y='longitude',  data=df_kmean,
           fit_reg=False, # No regression line
           hue='cluster',
           #, palette= pokemon_type_colors
           )
plt.title("clustering with k=2")
plt.show()

