from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

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

# Encoding categorical variables
data = pd.get_dummies(data, columns=['Month', 'VisitorType', 'Weekend'], drop_first=True)

'''
# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

'''
data_scaled=data

# True labels
true_labels = data['Revenue']

# Remove the 'Revenue' attribute
data = data.drop('Revenue', axis=1)

# K-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_clusters = kmeans.fit_predict(data_scaled)
kmeans_ri_score = adjusted_rand_score(true_labels, kmeans_clusters)

# Agglomerative clustering with complete-linkage
agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='complete')
agg_clusters = agg_cluster.fit_predict(data_scaled)
agg_ri_score = adjusted_rand_score(true_labels, agg_clusters)

print(f"K-means RI score: {kmeans_ri_score}")
print(f"Complete-Linkage AGNES RI score: {agg_ri_score}")

# Output which model is better
if kmeans_ri_score > agg_ri_score:
    print("K-means performs better in terms of RI score.")
elif kmeans_ri_score < agg_ri_score:
    print("Complete-Linkage AGNES performs better in terms of RI score.")
else:
    print("Both models perform equally in terms of RI score.")
