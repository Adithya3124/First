import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
age = df['age'].dropna()
print("Mean:", age.mean())
print("Median:", age.median())
print("Mode:", age.mode()[0])
print("Std Dev:", age.std())
print("Variance:", age.var())
print("Range:", age.max() - age.min())
age.plot.hist(title='Age Histogram')
plt.show()
age.plot.box(title='Age Boxplot')
plt.show()
Q1 = age.quantile(0.25)
Q3 = age.quantile(0.75)
IQR = Q3 - Q1
outliers = age[(age < Q1 - 1.5 * IQR) | (age > Q3 + 1.5 * IQR)]
print("\nOutliers:\n", outliers)
freq = df['class'].value_counts()
print("\nClass Frequencies:\n", freq)
freq.plot(kind='bar', title='Class Frequency')
plt.show()
