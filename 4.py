from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print("Regular k-NN:")
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"k={k} | Accuracy: {accuracy_score(y_test, y_pred):.2f}, F1: {f1_score(y_test, y_pred, average='macro'):.2f}")
print("\nWeighted k-NN (1/d²):")
for k in [1, 3, 5]:
    model = KNeighborsClassifier(n_neighbors=k, weights=lambda d: 1 / (d**2 + 1e-5))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"k={k} | Accuracy: {accuracy_score(y_test, y_pred):.2f}, F1: {f1_score(y_test, y_pred, average='macro'):.2f}")
