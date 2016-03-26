import numpy as np



def read_auto_data():
	filename = "auto-mpg.data"
	the_file = open(filename, "r")
	target_column = 8


def read_flare():
	filename = "flare.data"
	the_file = open(filename, "r")
	target_column = 0


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
	print(data)
	print(targets)


main()