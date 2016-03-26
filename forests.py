from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import read_file as rf

def main():
	data, targets = rf.read_letters()

	clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	scores = cross_val_score(clf, data, targets)
	print("Forrest_Letters: ", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_abalone()
	clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	scores = cross_val_score(clf, data, targets)
	print("Forrest_Abalone: ", end="")
	print(scores.mean() * 100)

"""
	data, targets = rf.read_lungs()
	clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	scores = cross_val_score(clf, data, targets)
	print("Forrest_Lungs: ", end="")
	print(scores.mean() * 100)
"""

main()


#iris = load_iris()

#clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1, random_state=0)
#scores = cross_val_score(clf, iris.data, iris.target)
#print(scores.mean() * 100)

#clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
#scores = cross_val_score(clf, iris.data, iris.target)
#print(scores.mean() * 100)

#clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
#scores = cross_val_score(clf, iris.data, iris.target)
#print(scores.mean() * 100)
