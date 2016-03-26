from sklearn.cross_validation import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def read_letters():
	filename = "letter-recognition.data"
	the_file = open(filename, "r")
	target_column = 0
	data = np.zeros((20000,16))
	targets = []
	row = 0


	for line in the_file:
		temp_string = line.split(",")
		targets.append(temp_string[0])
		temp_string.pop(0)
		column = 0
		for split in temp_string:
			split.rstrip('\n')
			data[row][column] = int(split)
			column += 1
		row += 1

	return data, targets

def main():
	data, targets = read_letters()

	clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=1, random_state=0)
	scores = cross_val_score(clf, data, targets)
	print(scores.mean() * 100)

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
