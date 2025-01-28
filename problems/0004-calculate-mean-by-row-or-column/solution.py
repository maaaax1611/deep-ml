def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:

	means = []
	if mode =='column': #calculate mean of column
		matrix = [list(row) for row in zip(*matrix)] #transpose 

	#if mode == 'row': #calculate mean of row
		#nothing happens

	for row in matrix:
		temp = 0
		for val in row:
			temp += val
		means.append(temp/len(row))

	return means