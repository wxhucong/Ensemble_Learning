from sklearn.cross_validation import cross_val_score
from sklearn import svm
import read_file as rf

def main():
	
	data, targets = rf.read_letters()
	clf = svm.SVC()
	scores = cross_val_score(clf, data, targets)
	print("SVM_Letters: ", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_abalone()
	clf = svm.SVC()
	scores = cross_val_score(clf, data, targets)
	print("SVM_Abalone: ", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_lungs()
	clf = svm.SVC()
	scores = cross_val_score(clf, data, targets)
	print("SVM_Lungs: ", end="")
	print(scores.mean() * 100)


main()