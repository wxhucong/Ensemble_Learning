from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import cross_val_score
import read_file as rf

def main():
	data, targets = rf.read_letters()
	clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(data)
	scores = cross_val_score(clf, data, targets)
	print("KNN_Letters: ", end="")
	print(scores.mean() * 100)

main()