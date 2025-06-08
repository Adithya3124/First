import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
X, y = fetch_california_housing(return_X_y=True)
model = LinearRegression().fit(X[:, [0]], y)
plt.scatter(X[:, 0], y, s=1)
plt.plot(X[:, 0], model.predict(X[:, [0]]), c='r')
plt.title("Linear Regression (MedInc vs Price)")
plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.show()
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ['mpg', 'cyl', 'disp', 'hp', 'wt', 'acc', 'yr', 'ori', 'name']
df = pd.read_csv(url, names=cols, sep=r'\s+', na_values='?').dropna()
X_poly = df[['hp']].astype(float).values
y_poly = df['mpg'].values
model_poly = make_pipeline(PolynomialFeatures(2), LinearRegression())
model_poly.fit(X_poly, y_poly)
y_pred = model_poly.predict(X_poly)
plt.scatter(X_poly, y_poly, c='blue', s=10, label='Actual')
plt.scatter(X_poly, y_pred, c='red', s=10, label='Predicted')
plt.title("Polynomial Regression (HP vs MPG)")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.legend()
plt.show()
