from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import read_file as rf

def main():
	data, targets = rf.read_letters()

	clf = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
	scores = cross_val_score(clf, data, targets)
	print(scores.mean() * 100)

main()

