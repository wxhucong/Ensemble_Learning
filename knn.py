from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
import read_file as rf

def main():
	data, targets = rf.read_letters()
	clf = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
	scores = cross_val_score(clf, data, targets)
	print("KNN_Letters: ", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_abalone()
	clf = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
	scores = cross_val_score(clf, data, targets)
	print("KNN_Abalone: ", end="")
	print(scores.mean() * 100)

	data, targets = rf.read_lungs()
	clf = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')
	scores = cross_val_score(clf, data, targets)
	print("KNN_Lungs: ", end="")
	print(scores.mean() * 100)

main()