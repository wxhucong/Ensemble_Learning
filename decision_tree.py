import read_file as rf
from sklearn import tree
from sklearn.cross_validation import cross_val_score

def main():

	data, targets = rf.read_letters()
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	scores = cross_val_score(clf, data, targets)
	print("DecisionTree_Letters", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_abalone()
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	scores = cross_val_score(clf, data, targets)
	print("DecisionTree_Abalone", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_lungs()
	clf = tree.DecisionTreeClassifier(criterion='entropy')
	scores = cross_val_score(clf, data, targets)
	print("DecisionTree_Lungs", end="")
	print(scores.mean() * 100)


main()