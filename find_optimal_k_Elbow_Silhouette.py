"""
find the optimal k
1. using elbow method
using the distortion method to find the optimal k, (elbow point).
The inertia is the squared error of a point and its cluster,
the inertia is a similarity measure of a point to its centroide (distance between a point and its centrode)
When the similarity (interia) decrease is good because we classify the point correctly.
The distortion is the average of inertia of all points (observation).
When the distortion is reduce is better.

2. using silhouette score

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
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer  #get the optimal k using elbow
from sklearn.metrics import silhouette_score, silhouette_samples #find the optimal k using sihouette score
import pdb

url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)

df_kmean=df[['latitude','longitude']].copy()
# Instantiate the clustering model and visualizer


"""
finding optimal k using silhouette score
"""
# silhouette_score={ }
k_range=w=list(range(4,20)) #specify the range of k
for k in k_range:
    clusterer= KMeans(n_clusters=k, init='k-means++', random_state=20)
    cluster_labels=clusterer.fit_predict(df_kmean.values)#Compute cluster centers and predict cluster index for each sample
    #print(cluster_labels)
    #the silhouette score gives the av. value for all the samples
    #this gives a percepective  into density and seperation of the formed cluster
    silhouette_avg=silhouette_score(df_kmean.values,cluster_labels )
    #silhouette_score[k]=silhouette_avg #to store the k and its corresponding silhouette score
    print(f"for k cluster={k}, the av silhouette_score is {silhouette_avg}") #we can see that k>= 13, we get good score

#I will use k=15 as the elbow method, because when I compared the dissimilarity and similarity perforamce
# , using optimal_k_compare, I saw thatk=15 gives better result
"""
finding optimal k using visualization (elbow)
"""s
model = KMeans(init='k-means++')
visualizer = KElbowVisualizer(model, k=(4,100), timings=False) #k is th enumber of cluter
visualizer.fit(df_kmean.values)     # Fit the data to the visualizer
visualizer.show()                   # Finalize and render the figure



