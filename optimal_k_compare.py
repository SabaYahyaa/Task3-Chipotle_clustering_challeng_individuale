"""
I will compare k=2 and k=15 to see if I get better performance according to dissimilarity and inertia
K mean, k=2
x='latitude', y='longitude',
Note: we do not need to standarize our data since features have the same scales

Note; when you run the program, you will get 3 graphs and
------------------ k=2 -----------------------------------
The av_inertia for all samples
 197651.70088830453
The av of dissimilarity for samples
 33.62503103723478
------------------ k=13 -----------------------------------
The av_inertia for all samples
 11054.643533114355
The av of dissimilarity for samples
 268.8062425962732
------------------ k=15 -----------------------------------
The av_inertia for all samples
 8414.91540725273
The av of dissimilarity for samples
 303.1729997776526
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


"""
k=2
"""
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
Note: I will compare the average similarity (distortion) and average dissimilarity to see if there is an improvement
"""
print("------------------ k=2 -----------------------------------")
#get the av inertia (distortion of each cluster) of each cluster
av_inertia_samples=df_kmean["Inertia"].mean()
print(f"The av_inertia for all samples \n {av_inertia_samples}")

#get the average dissimilarity for each class
av_dis=df_kmean["Dissimilarities"].mean()
print(f"The av of dissimilarity for samples\n {av_dis}")

"""
k=13, this is the optimal value according  to Silhouette score
"""
df_kmean_13=df[['latitude','longitude']].copy()

# number of pokemon clusters
state_size = 13#'k-means++' is to initiate the centrodic in a smart way, to get faster convergence
kmeans_13=KMeans(n_clusters=state_size, init='k-means++').fit(df_kmean_13.values)



# add cluster index to dataframe, this will e your observation
df_kmean_13['cluster']=kmeans_13.labels_

#compute the dissimilarities using Euclidean distance and create a new column in the df
all_observation=df_kmean_13.values
# clusters is an attribute of the object
cluster_centers = kmeans_13.cluster_centers_
#compute the dissimilarities
diss=dissimilarities(all_observation, cluster_centers, distance_type='Eculidean')

#add the dis-similarity
df_kmean_13["Dissimilarities"]=diss

#add the inertia
df_kmean_13["Inertia"]=kmeans_13.inertia_


"""
Note: I will compare the average similarity (distortion) and average dissimilarity to see if there is an improvement
"""
print("------------------ k=13 -----------------------------------")
#get the av inertia (distortion of each cluster) of each cluster
av_inertia_samples=df_kmean_13["Inertia"].mean()
print(f"The av_inertia for all samples \n {av_inertia_samples}")

#get the average dissimilarity for each class
av_dis=df_kmean_13["Dissimilarities"].mean()
print(f"The av of dissimilarity for samples\n {av_dis}")




"""
k=15, according to elbow curve, the best k=15
this is the optimal value according to elbow curve
"""
df_kmean_15=df[['latitude','longitude']].copy()

# number of pokemon clusters
state_size = 15#'k-means++' is to initiate the centrodic in a smart way, to get faster convergence
kmeans_15=KMeans(n_clusters=state_size, init='k-means++').fit(df_kmean_15.values)



# add cluster index to dataframe, this will e your observation
df_kmean_15['cluster']=kmeans_15.labels_

#compute the dissimilarities using Euclidean distance and create a new column in the df
all_observation=df_kmean_15.values
# clusters is an attribute of the object
print(cluster_centers )
#compute the dissimilarities
diss=dissimilarities(all_observation, cluster_centers, distance_type='Eculidean')

#add the dis-similarity
df_kmean_15["Dissimilarities"]=diss

#add the inertia
df_kmean_15["Inertia"]=kmeans_15.inertia_

"""
Note: I will compare the average similarity (distortion) and average dissimilarity to see if there is an improvement
"""
print("------------------ k=15 -----------------------------------")
#get the av inertia (distortion of each cluster) of each cluster
av_inertia_samples=df_kmean_15["Inertia"].mean()
print(f"The av_inertia for all samples \n {av_inertia_samples}")

#get the average dissimilarity for each class
av_dis=df_kmean_15["Dissimilarities"].mean()
print(f"The av of dissimilarity for samples\n {av_dis}")







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


# plot data with the type color palette
sns.lmplot(x='latitude', y='longitude',  data=df_kmean_13,
           fit_reg=False, # No regression line
           hue='cluster',
           #, palette= pokemon_type_colors
           )
plt.title("clustering with k=13")


# plot data with the type color palette
sns.lmplot(x='latitude', y='longitude',  data=df_kmean_15,
           fit_reg=False, # No regression line
           hue='cluster',
           #, palette= pokemon_type_colors
           )
plt.title("clustering with k=15")


plt.show()

