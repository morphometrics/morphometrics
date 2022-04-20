import napari
import numpy as np

from morphometrics.data import simple_labeled_cube
from morphometrics.measure import available_measurments

print(f"available_measurements: {available_measurments()}")

label_image = simple_labeled_cube()
intensity_image = np.random.random(label_image.shape)

viewer = napari.Viewer()
viewer.add_image(intensity_image)
viewer.add_labels(label_image)

# to use the plugin, select plugins -> morphometrics: Make region properties measurement widget

# measurement_widget = QtMeasurementWidget(viewer)
# viewer.window.add_dock_widget(widget=measurement_widget)

napari.run()
