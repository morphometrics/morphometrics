import napari
import numpy as np
from qtpy.QtGui import QCursor

from morphometrics._gui._qt.radial_menu import ButtonSpecification, RadialMenu

viewer = napari.Viewer()
image_layer = viewer.add_image(np.random.rand(30, 30))


def example_action():
    print("hello")


button_list = [ButtonSpecification(name="test", action=example_action)]


@viewer.mouse_drag_callbacks.append
def mousePressEvent(viewer, event):
    # this is automatically called when a mouse key press event occurs
    if event.button == 2:
        global_position = QCursor.pos()
        canvas_widget = viewer.window._qt_window._qt_viewer.canvas.native
        canvas_widget_position = canvas_widget.mapFromGlobal(global_position)
        menu = RadialMenu(canvas_widget, canvas_widget_position, buttonList=button_list)
        menu.show()


if __name__ == "__main__":
    napari.run()
