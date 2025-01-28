def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
	
	if len(a[0]) != len(b): #matrix and vector cannot be dotted
		return -1
	
	c = [] #init empty list
	for row in a: #row containts the values from the current row
		value = 0
		for i in range(len(row)): #iterate over the values from the current row
			value += row[i]*b[i]
		c.append(value)
		
	return c