from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.ensemble import BaggingClassifier as Bag, AdaBoostClassifier as Boost
from sklearn.metrics import accuracy_score as acc, classification_report as rep
X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = tts(X, y, test_size=0.3, random_state=42)
bag = Bag(DT(), n_estimators=50, random_state=42)
boost = Boost(DT(max_depth=1), n_estimators=50, random_state=42)
bag.fit(Xtr, ytr)
boost.fit(Xtr, ytr)
pb = bag.predict(Xte)
pb2 = boost.predict(Xte)
print("Bagging Accuracy:", acc(yte, pb))
print("Boosting Accuracy:", acc(yte, pb2))
print("\nBagging Report:\n", rep(yte, pb))
print("\nBoosting Report:\n", rep(yte, pb2))
3
