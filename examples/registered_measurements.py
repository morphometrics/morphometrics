from morphometrics.data import simple_labeled_cube
from morphometrics.measure import available_measurments, measure_all

print(f"available_measurements: {available_measurments()}")

label_im = simple_labeled_cube()

measurement_table = measure_all(label_im)
print(measurement_table)
