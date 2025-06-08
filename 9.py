from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
data = load_breast_cancer()
X = data.data
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(X)
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='coolwarm')
plt.title("K-Means Clustering on Breast Cancer Data")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.show()
