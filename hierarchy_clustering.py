"""
this program, uses  hierarchical clustering, AgglomerativeClustering and draw the dendogram curve
we need to find the optimal parameter for each clustering model, then we compare between them

DBSCAN
https://medium.com/@mohantysandip/a-step-by-step-approach-to-solve-dbscan-algorithms-by-tuning-its-hyper-parameters-93e693a91289
and
https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html

"""
from sklearn.cluster import KMeans
from sklearn import preprocessing #we will normalize our data
import pdb
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg as sp
from sklearn.model_selection import train_test_split
import seaborn as sns


url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)

df_sub=df[['latitude','longitude']].copy()
###########################################
###########################################
###########################################
#       hierarchical clustering
#https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318
###########################################
###########################################
###########################################

###########################################
######  draw the dendogram, to find the number of clustering
######  find the largest vetrical distance without cutting the horizontal line
###########################################
#Using the dendrogram to find the optimal numbers of clusters.
import scipy.cluster.hierarchy as sch

#specify the likage measure between clusters
dendrogram = sch.dendrogram(sch.linkage(df_sub.values, method  = "single"))
plt.title('Dendrogram using single Leakage')
plt.xlabel('Locations')
plt.ylabel('Euclidean distances')
#plt.show()
#single: 	Performs single/min/nearest linkage on the condensed distance matrix y
# complete:	Performs complete/max/farthest point linkage on a condensed distance matrix
# average: 	Performs average/UPGMA linkage on a condensed distance matrix
# weighted: 	Performs weighted/WPGMA linkage on the condensed distance matrix.
# centroid: 	Performs centroid/UPGMC linkage.
# median: 	Performs median/WPGMC linkage.
# ward:    	Performs Ward’s linkage on a condensed or redundant distance matrix.

# Ward method tries to minimize the variance within each cluster.
# In K-means when we were trying to find the min distortion to plot the elbow
# ward is almost the same the only difference is that instead of minimizing distortion
# we are minimizing the within-cluster variants.
# That is the variance within each cluster.

# How do we determine the optimal number of clusters from this diagram?
# We look for the largest distance that we can vertically without crossing any horizontal line
# after applying many leakage, I find ward gives better splitting


###########################################
######  fitting using sklearn
###########################################
# There are two algorithms for hierarchical clustering:
# #Agglomerative and Divisive Hierarchical Clustering.
# We choose Euclidean distance and ward method for our algorithm class
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage ='ward')

cluster=hc.fit_predict(df_sub)

df_sub['cluster']=cluster

"""
visualization
"""
# plot data with the type color palette
sns.lmplot(x='latitude', y='longitude',  data=df_sub,
           fit_reg=False, # No regression line
           hue='cluster',
           #, palette= pokemon_type_colors
           )
plt.title("clustering with k=2")
plt.show()
