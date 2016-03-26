from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)

scores = cross_val_score(bagging, iris.data, iris.target)
print(scores.mean() * 100)