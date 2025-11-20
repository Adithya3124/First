import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = pd.read_csv('zoo.csv')
X = data.select_dtypes(include='number')
data['Cluster'] = KMeans(n_clusters=3, random_state=42).fit_predict(X)
print(data.head())
if X.shape[1] == 2:
    plt.scatter(X.iloc[:,0], X.iloc[:,1], c=data['Cluster'], cmap='viridis')
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title('K-Means Clustering')
    plt.show()
4
