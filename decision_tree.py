import read_file as rf
from sklearn import tree
from sklearn.cross_validation import cross_val_score

data, targets = rf.read_abalone()
clf = tree.DecisionTreeClassifier(criterion='entropy')
scores = cross_val_score(clf, data, targets)
print(scores.mean() * 100)













