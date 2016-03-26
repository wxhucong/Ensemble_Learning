from sklearn.cross_validation import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
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

	clf = AdaBoostClassifier(n_estimators=100, learning_rate=.007)
	scores = cross_val_score(clf, data, targets)
	print(scores.mean() * 100)
	#print(data)
	#print(targets)


main()

#iris = load_iris()
#clf = AdaBoostClassifier(n_estimators=100)
#scores = cross_val_score(clf, iris.data, iris.target)
#print(scores.mean() * 100)