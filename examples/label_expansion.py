import napari
import numpy as np

from morphometrics._gui._qt.labeling_widgets import QtLabelingWidget

label_image = np.zeros((100, 100, 100), dtype=int)
label_image[30:70, 30:70, 30:70] = 1
label_image[45:55, 45:55, 45:55] = 0

label_image[49:51, 49:51, 49:51] = 2

viewer = napari.Viewer(ndisplay=3)
viewer.add_labels(label_image)

labeling_widget = QtLabelingWidget(viewer)
viewer.window.add_dock_widget(labeling_widget)

napari.run()
