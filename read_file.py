import numpy as np



def read_lungs():
	filename = "lung-cancer.data"
	the_file = open(filename, "r")
	target_column = 0
	data = np.zeros((32,56))
	targets = []
	row = 0

	for line in the_file:
		temp_string = line.split(",")
		targets.append(temp_string[target_column].rstrip('\n'))
		temp_string.pop(target_column)
		column = 0
		for split in temp_string:
			if split == "?":
				if column == 3:
					split = "-1"
				else:
					split = "4"
			data[row][column] = float(split)
			column += 1
		row += 1

	return data, targets



def read_abalone():
	filename = "abalone.data"
	the_file = open(filename, "r")
	target_column = 0
	data = np.zeros((4177,8))
	targets = []
	row = 0

	for line in the_file:
		temp_string = line.split(",")
		targets.append(temp_string[8].rstrip('\n'))
		temp_string.pop(8)
		column = 0
		for split in temp_string:
			split.rstrip('\n')
			if column == 0:
				if split == 'M':
					split = "0"
				elif split == 'F':
					split = "1"
				else:
					split = "2"

			data[row][column] = float(split)
			column += 1
		row += 1

	return data, targets


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
	data, targets = read_lungs()
	print(data)
	print(targets)


main()