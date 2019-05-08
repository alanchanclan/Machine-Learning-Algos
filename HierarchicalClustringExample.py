import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/cars_clus.csv

pdf = pd.read_csv('cars_clus.csv')
print("Shape of dataset: ", pdf.shape)
print(pdf.head(5))

print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce') # if error, set it to NaN
pdf = pdf.dropna() # drop the rows with missing or NaN values
pdf = pdf.reset_index(drop=True)
print("Shape of dataset after cleaning: ", pdf.size)
print(pdf.head(5))

featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

# Normalize the data
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler() # Min Max scaler scales all values to a specific range by default is 0 - 1
feature_mtx = min_max_scaler.fit_transform(x) # Each list in the multidimensional array is a row in pd dataframe
print(feature_mtx[0:5])

##### USING SCIPY #####

# Calculate the distance matrix
import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng]) # Creates leng by leng numpy array filling in the remainder with 0's
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j]) # the distance of everything

# There exists 5 methods to calculate distance
# - single
# - complete
# - average
# - weighted
# - centroid

import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance') # cutting line, max distance
print(clusters)

# Specify the number of clusters directly
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

# Create a dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )

dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')
plt.show()
plt.close()

##### USING SCIKIT-LEARN #####

agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
print(agglom.labels_)

pdf['cluster_'] = agglom.labels_
pdf.head()

import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25)
    color_temp = [[]]
    color = color_temp[0].append(color)
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
plt.close()

print(pdf.groupby(['cluster_','type'])['cluster_'].count())
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean() # changes order of list, may be handy for using numpy .index
print(agg_cars)

# It is obvious that we have 3 main clusters with the majority of vehicles in those.
#
# Cars:
#
# Cluster 1: with almost high mpg, and low in horsepower.
# Cluster 2: with good mpg and horsepower, but higher price than average.
# Cluster 3: with low mpg, high horsepower, highest price.

# Trucks:
#
# Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
# Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
# Cluster 3: with good mpg and horsepower, low price.

plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),] # access the whole row with the label 0.0 or 1.0 (, groups columns), label is removed
    for i in subset.index: # returns 1.0 and 0.0 for cars and trucks
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k') # horsepow, mpg, type, price in k
    color_temp = [[]]
    color = color_temp[0].append(color)
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
