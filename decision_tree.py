from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
iris = load_iris()
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(iris.data, iris.target)
print("Iris accuracy: " + str(clf.score(iris.data, iris.target)) + "%")













