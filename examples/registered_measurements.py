import numpy as np

from morphometrics.measure import available_measurments, measure_all

print(f"available_measurements: {available_measurments()}")

label_im = np.zeros((10, 10, 10), dtype=int)
label_im[5:10, 5:10, 5:10] = 1
label_im[5:10, 0:5, 0:5] = 2
label_im[0:5, 0:10, 0:10] = 3

measurement_table = measure_all(label_im)
print(measurement_table)
