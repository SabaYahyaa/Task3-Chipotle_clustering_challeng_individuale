"""
this program produces USA map,
uses abbreviation and put the number of appearance of each state at the map
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


url='https://raw.githubusercontent.com/becodeorg/BXL-Bouman-2.22/master/content/04.machine_learning/3.Clustering/assets/chipotle_stores.csv?token=AHYQ2CVQ7AR2USNRZD42OUS74XIIY'
df=pd.read_csv(url)

#print(df.columns.tolist()) #['state', 'location', 'address', 'latitude', 'longitude']
#Since I will not use address, I will delete it
del df['address']


##############################################################################
####    draw USA map in python
# git clone https://jcutrer.com/python/learn-geopandas-plotting-usmaps
###################################################################################
import matplotlib.pyplot as plt
import geopandas
import pdb


#read the following file
states = geopandas.read_file('/home/saba/PycharmProjects/testing/Machine_Learning/Clustering/Challenge_Clustering/geopandas-tutorial/data/usa-states-census-2014.shp')
# print(type(states))
#
# print(states.head())
# print(states.columns.tolist())
# print(states.shape)
#print(states.NAME.value_counts())
print("------------------------------------------------")
#print(df.state.value_counts())
# # #specify a method for appearance

#get the names of state from df,
df_states_name=dict(df.state.value_counts()).keys()

#get the states_NAME
state_NAME=dict(states.NAME.value_counts()).keys()

#instead of using the name of a state, I want the first 3 letter if the state does not contains space,
# else I get only the first letters of the state name
New_Name={}
for s in state_NAME:
    if s in New_Name:
        pass
    else:
        for ss in df_states_name:
            if ss==s:
                if ' ' not in ss:
                    New_Name[s]=ss[0:3] #if there is no space, select only the first 3 letters
                else:
                    words = ss.split() #if there is a space, select the first letter of each word
                    letters = [word[0] for word in words]
                    ss = "".join(letters)
                    New_Name[s]=ss

#this is the output
# New_Name={"Pennsylvania":  "Pen"  ,
# "New York":"NY",
# "Rhode Island":"RI",
# "New Jersey": "NJ"  ,
# "Massachusetts":"Mas",
# "New Hampshire": "NH" ,
# "Connecticut":  "Con"  ,
# "Maine":   "Mai"   ,
# "Vermont":  "Ver",
# "Kentucky": "Ken" ,
# "New Mexico":  "NM",
# "Maryland":"Mar",
# "Illinois": "Ill",
# "Missouri" :  "Mis",
# "Indiana": "Ind"  ,
# "Washington":  "Was",
# "Texas": "Tex" ,
# "Arkansas": "Ark",
# "Wyoming": "Wyo"  ,
# "Delaware": "Del"  ,
# "West Virginia": "WV",
# "Ohio":  "Ohi" ,
# "Arizona":  "Ari" ,
# "Minnesota": "Min",
# "Idaho":   "Ida",
# "Wisconsin" : "Wis",
# "Iowa":  "Iow" ,
# "Florida": "Flo",
# "Kansas":    "Kan",
# "North Dakota":  "ND",
# "District of Columbia": None,
# "Alabama":    "Ala",
# "California":  "Cal",
# "Nebraska": "Neb",
# "Oklahoma":  "Okl",
# "Nevada":  "Nev",
# "Georgia": "Geo",
# "Michigan": "Mic",
# "Tennessee":  "Ten"  ,
# "South Carolina": "SC",
# "South Dakota":        None,
# "North Carolina": "NC" ,
# "Mississippi":   "Mis"  ,
# "Oregon" :   "Ore"       ,
# "Montana":    "Mon"       ,
# "Virginia":  "Vir" ,
# "Colorado" :"Col"   ,
# "Utah":        "Uta" ,
# "Louisiana": "Lou" }


#
# #get the number of apperance of each state in our df
count_states_in_df=dict(df.state.value_counts())
count_states=(states.NAME.map(count_states_in_df))

#I got an error, so I did that
def to_string(x):
    try:
        if x !=None:
            x=str(int(x))
            return (x)
        else:
            return (" ")
    except ValueError:
        return (" ")

states["count_apperance"]=(count_states) #create a new column in states
states["count_apperance"]=states["count_apperance"].apply(lambda x: to_string(x))

print(states["count_apperance"])

# print(count_states)
states.NAME=states.NAME.map(New_Name)


# # print(state_NAME)
# # # 33333333333333333333333333333333333333333333333333333""
# # #draw the USA map
states = states.to_crs("EPSG:3395")
# #Plotting Shapefiles
fig = plt.figure(1, figsize=(25,25))
ax = fig.add_subplot()
states.apply(lambda x: ax.annotate(s=str(x.NAME) +"\n" + str(x.count_apperance), xy=x.geometry.centroid.coords[0],ha='center', fontsize=8),axis=1)
states.boundary.plot(ax=ax, color='Black', linewidth=.4)
ax.set(title="USA, using abbreviation for states and \n the number of appearance of each state in our data")
states.plot(ax=ax, cmap='Pastel2', figsize=(25, 25))


# # # 33333333333333333333333333333333333333333333333333333333333333333333333333""



# #draw the boundary
# states.boundary.plot()
# #draw USA map with color
# states.plot(cmap='magma', figsize=(12, 12))
# #drawing one state, e.g. Taxes
# states[states['NAME'] == 'Texas'].plot(figsize=(12, 12))
# # Plotting multiple shapes
# # Plot multiple states together, e.g. stattes that makeup the South East region of the USA.
# #1. select states
# southeast = states[states['STUSPS'].isin(['FL','GA','AL','SC','NC', 'TN', 'AR', 'LA', 'MS'])]
# #2.plot the selected states
# southeast.plot(cmap='tab10', figsize=(14, 12))
# #plot by region
# southeast = states[states['region'] == 'Southeast']
# southeast .plot()
# #west region
# west = states[states['region'] == 'West']
# west.plot(cmap='Pastel2', figsize=(12, 12))
#
# # us_boundary_map = states.boundary.plot(figsize=(18, 12), color="Gray")
# # sns.lmplot(x='latitude', y='longitude',  data=df,
# #            fit_reg=False, # No regression line
# #            #, palette= pokemon_type_colors
# #            )



plt.show()





