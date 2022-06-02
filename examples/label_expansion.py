import napari
import numpy as np

label_image = np.zeros((100, 100, 100), dtype=int)
label_image[30:70, 30:70, 30:70] = 1
label_image[45:55, 45:55, 45:55] = 0

label_image[49:51, 49:51, 49:51] = 2

background_mask = np.ones_like(label_image, dtype=bool)
background_mask[30:70, 30:70, 30:70] = 0

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(background_mask)
viewer.add_labels(label_image)
viewer.window.add_plugin_dock_widget(
    plugin_name="morphometrics", widget_name="Curate labels"
)

napari.run()
