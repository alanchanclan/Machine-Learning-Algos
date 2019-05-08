import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv

import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
print(cust_df.head())

df = cust_df.drop('Address', axis=1) # axis 1 is y axis, we drop this row bc it is discrete and non-numerical
print(df.head())

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:] # Takes out the first column
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
print(Clus_dataSet)

clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)

df["Clus_km"] = labels # add column concerning their classification
df.head(5)

df.groupby('Clus_km').mean() # find centroid for datas

area = np.pi * ( X[:, 1])**2  # Uses education column
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))
plt.show()
# We can now create profiles for each group, such as:
# -> AFFLUENT, EDUCATED AND OLD AGED
# -> MIDDLE AGED AND MIDDLE INCOME
# -> YOUNG AND LOW INCOME