import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('online_shoppers_intention.csv')

# Data preprocessing
# Numericalize 'Weekend' and 'Revenue'
data['Weekend'] = data['Weekend'].map({False: 0, True: 1})
data['Revenue'] = data['Revenue'].map({False: 0, True: 1})

# Mean Encoding for 'Month'
month_means = data.groupby('Month')['Revenue'].mean()
data['Month'] = data['Month'].map(month_means)

# Mean Encoding for 'VisitorType'
visitortype_means = data.groupby('VisitorType')['Revenue'].mean()
data['VisitorType'] = data['VisitorType'].map(visitortype_means)

# Remove the 'Revenue' attribute
data = data.drop('Revenue', axis=1)

# Encoding categorical variables
data = pd.get_dummies(data, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)

# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Applying agglomerative clustering with complete-linkage
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='complete')
data['Cluster'] = agg_cluster.fit_predict(data_scaled)

# Visualizing the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Administrative', y='PageValues', hue='Cluster', data=data, palette='Set1')
plt.title('Clusters of Online Shoppers')
plt.xlabel('Administrative')
plt.ylabel('PageValues')
plt.show()

# Interpret the results
cluster_means = data.groupby('Cluster').mean()
print(cluster_means)
