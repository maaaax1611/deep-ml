import math
import numpy as np

PI = 3.14159

def detrend_data(data):
    return [x - 10 * math.sin(2 * PI * i / 10) for i, x in enumerate(data, start=1)]

def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	detrended = detrend_data(consumption_data)
	# 2) Perform linear regression on the detrended data.
	days = np.arange(1, len(detrended)+1)
	m, b = np.polyfit(days, detrended, 1)
	# 3) Predict day 15's base consumption.
	pred_15 = m * 15 + b
	# 4) Add the day 15 fluctuation back.
	total_15 = pred_15 + 10 * math.sin(2 * PI * 15 / 10)
	# 5) Round, then add a 5% safety margin (rounded up).
	total_15 = int(math.ceil(round(total_15) * 1.05))
	# 6) Return the final integer.
	return total_15