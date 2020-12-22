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
import geopandas


url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)
print(df.columns.tolist())
#get the geometry of USA
# gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.longitude, df.latitude))
#
# print(gdf.head())
# fig, ax = plt.subplots()
# #this is without any clustering
# world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
# # We restrict to South America.
# usa = world[world.name == 'United States'].plot(ax=ax, color='white', edgecolor='black')
# # We can now plot our ``GeoDataFrame``.
# gdf.plot(ax=usa, color='red')
#
# plt.ylabel('Latitude')
# plt.xlabel('Longitude')


### use kmean with 3 clusters,

# number of pokemon clusters
state_size = 3
kmeans=KMeans(n_clusters=state_size, init='k-means++').fit(df[['longitude', 'latitude']].values)
# add cluster index to dataframe, this will e your observation
df['cluster']=kmeans.labels_

gb=df.groupby('cluster')
cluster1=gb.get_group(1)
cluster2=gb.get_group(2)
cluster3=gb.get_group(0)

gdf1 = geopandas.GeoDataFrame(cluster1, geometry=geopandas.points_from_xy(cluster1.longitude,cluster1.latitude))
gdf2 = geopandas.GeoDataFrame(cluster2, geometry=geopandas.points_from_xy(cluster2.longitude, cluster2.latitude))
gdf3 = geopandas.GeoDataFrame(cluster3, geometry=geopandas.points_from_xy(cluster3.longitude, cluster3.latitude))

fig, ax = plt.subplots()
#this is without any clustering
world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

usa = world[world.name == 'United States'].plot(ax=ax)
gdf1.plot(ax=usa, color='red',  scheme='quantiles', edgecolor='k', k=10, legend=True)
gdf2.plot(ax=usa, color='green', scheme='quantiles', edgecolor='k', k=10, legend=True)
gdf3.plot(ax=usa, color='blue' ,scheme='quantiles', edgecolor='k', k=10, legend=True)
plt.show()