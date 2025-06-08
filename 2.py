import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import pearsonr
df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.title('Sepal vs Petal Length')
plt.show()
r, _ = pearsonr(df['sepal length (cm)'], df['petal length (cm)'])
print("Pearson Correlation:", round(r, 2))
print("\nCovariance:\n", df.cov())
print("\nCorrelation:\n", df.corr())
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Heatmap")
plt.show()
