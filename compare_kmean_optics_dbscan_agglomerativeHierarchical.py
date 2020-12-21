"""

1. I will compare OPTICS, k-mean, DBSCAN, Hierarchy agglomerative clustering algorithm
2. I will use the optimal parameters for each algorithm:
kmean (k=15 this is optimal value according to silhouette score and elbow)
DBSCAN (min_sample=31, eps=0.26)
OPTICS (min_sample=31 as DBSCAN)
Hierarchy agglomerative (hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward') according to dendogram)

3. my performance measure is the dissimilarity measure
I need to compute the labels, the the centroide for each class then compute the dissimilarity and similarity measure
"""
# #### output
# ------------------ Hierarchical Agglomerrative -----------------------------------
# The av of dissimilarity for samples using Hierarchical is 92.49760881627664
# ------------------ Optics -----------------------------------
# The av of dissimilarity for samples using OPTICS is 505.0906716983576
# ------------------ DBSCAN -----------------------------------
# The av of dissimilarity for samples using DBSCAN is 300.215624263335
# ------------------ k=15 -----------------------------------
# The av of dissimilarity for samples using K-mean is 303.22792885004594


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
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples #find the optimal k using sihouette score
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS

#compute the dissimilarities between each observation and all the centroide except the centroide of that oservation
def dis_similarity(observations, centroides):
    dissimilarities=[]
    for observation in observations:
        dist=0
        for c in centroides:
            if observation[-1]==c[-1]: #we have the same centroide, do not compute how far the observation from centroide
                   pass
            else:
                dist +=np.linalg.norm(c[0:-1] - observation[0:-1])
        dissimilarities.append(dist)
    return (dissimilarities)

url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)


"""
Hierarchical agglomerative
"""
df_hier=df[['latitude','longitude']].copy()
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward').fit(df_hier.values)
df_hier['label_hier']=hc.labels_

#compute the centroide
lat=df_hier.groupby('label_hier')['latitude'].mean().reset_index(name='centroide_latitude')
lon=df_hier.groupby('label_hier')['longitude'].mean().reset_index(name='centroide_longitude')
#store the centroides in a df
centroide_hier=pd.DataFrame({"cen_latitude": lat["centroide_latitude"],"cen_longitude": lon["centroide_longitude"], "label_hier": lat["label_hier"],})
#create a matrix
centroide_hier=centroide_hier.values
observation_hier=df_hier.values
#get the dissimilarity measure
df_hier['Dissimilarities']=dis_similarity(observation_hier, centroide_hier)

print("------------------ Hierarchical Agglomerrative -----------------------------------")
#get the average dissimilarity for each class
av_dis=df_hier['Dissimilarities'].mean()
print(f"The av of dissimilarity for samples using Hierarchical is {av_dis}")



"""
OPTICS
"""
df_optics=df[['latitude','longitude']].copy()
optics=OPTICS(min_samples=31).fit(df_optics.values)
df_optics['label_optics']=optics.labels_

#compute the centroide
lat=df_optics.groupby('label_optics')['latitude'].mean().reset_index(name='centroide_latitude')
lon=df_optics.groupby('label_optics')['longitude'].mean().reset_index(name='centroide_longitude')
#store the centroides in a df
centroide_optics=pd.DataFrame({"cen_latitude": lat["centroide_latitude"],"cen_longitude": lon["centroide_longitude"], "label_optics": lat["label_optics"],})
#create a matrix
centroide_optics=centroide_optics.values
observation_dbscan=df_optics.values
#get the dissimilarity measure
df_optics['Dissimilarities']=dis_similarity(observation_dbscan, centroide_optics)

print("------------------ Optics -----------------------------------")
#get the average dissimilarity for each class
av_dis=df_optics['Dissimilarities'].mean()
print(f"The av of dissimilarity for samples using OPTICS is {av_dis}")


"""
DBSCAN
"""
df_dbscan=df[['latitude','longitude']].copy()
db=DBSCAN(eps=0.25, min_samples=31).fit(df_dbscan.values)
labels_dbscan=db.labels_
df_dbscan['label_dbscan']=labels_dbscan

#compute the centroide
lat=df_dbscan.groupby('label_dbscan')['latitude'].mean().reset_index(name='centroide_latitude')
lon=df_dbscan.groupby('label_dbscan')['longitude'].mean().reset_index(name='centroide_longitude')
#store the centroides in a df
centroide_dbscan=pd.DataFrame({"cen_latitude": lat["centroide_latitude"],"cen_longitude": lon["centroide_longitude"], "label_dbscan": lat["label_dbscan"],})
#create a matrix
centroide_dbscan=centroide_dbscan.values
observation_dbscan=df_dbscan.values
#get the dissimilarity measure
df_dbscan['Dissimilarities']=dis_similarity(observation_dbscan, centroide_dbscan)

print("------------------ DBSCAN -----------------------------------")
#get the average dissimilarity for each class
av_dis=df_dbscan['Dissimilarities'].mean()
print(f"The av of dissimilarity for samples using DBSCAN is {av_dis}")



"""
k=15, according to elbow curve, the best k=15
this is the optimal value according to elbow curve
"""
df_kmean_15=df[['latitude','longitude']].copy()

# number of pokemon clusters
state_size_optimal = 15#'k-means++' is to initiate the centrodic in a smart way, to get faster convergence
kmeans_15=KMeans(n_clusters=state_size_optimal, init='k-means++').fit(df_kmean_15.values)
# add cluster index to dataframe, this will e your observation
df_kmean_15['cluster']=kmeans_15.labels_
#compute the dissimilarities using Euclidean distance and create a new column in the df
all_observation=df_kmean_15.values
# clusters is an attribute of the object
cluster_centers = kmeans_15.cluster_centers_
#compute the dissimilarities
diss=dissimilarities(all_observation, cluster_centers, distance_type='Eculidean')

#add the dis-similarity
df_kmean_15["Dissimilarities"]=diss

"""
Note: I will compare the average similarity (distortion) and average dissimilarity to see if there is an improvement
"""
print("------------------ k=15 -----------------------------------")
#get the average dissimilarity for each class
av_dis=df_kmean_15["Dissimilarities"].mean()
print(f"The av of dissimilarity for samples using K-mean is {av_dis}")
