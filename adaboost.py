from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import read_file as rf


def main():
	data, targets = rf.read_letters()
	clf = AdaBoostClassifier(n_estimators=100, learning_rate=.007)
	scores = cross_val_score(clf, data, targets)
	print("Adaboost_Letters", end="")
	print(scores.mean() * 100)


	data, targets = rf.read_abalone()
	clf = AdaBoostClassifier(n_estimators=100, learning_rate=.007)
	scores = cross_val_score(clf, data, targets)
	print("Adaboost_Abalone: ", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_lungs()
	clf = AdaBoostClassifier(n_estimators=100, learning_rate=.007)
	scores = cross_val_score(clf, data, targets)
	print("Adadboost_Lungs: ", end="")
	print(scores.mean() * 100)


main()