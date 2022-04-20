import anndata
import napari
import numpy as np

from morphometrics._gui._qt.annotation_widgets import QtClusterAnnotatorWidget
from morphometrics.data import cylinders_and_spheres
from morphometrics.explore.cluster import cluster_features
from morphometrics.explore.dimensionality_reduction import pca, umap
from morphometrics.explore.sample import sample_anndata
from morphometrics.measure import measure_selected
from morphometrics.utils.anndata_utils import table_to_anndata

rng = np.random.default_rng(42)

load_measurements_from_disk = True

# load the sample images
label_image, label_table, intensity_image = cylinders_and_spheres()


if load_measurements_from_disk is False:
    # make the measurements
    measurement_selection = [
        "surface_properties_from_labels",
        {
            "name": "regionprops",
            "choices": {
                "size": False,
                "intensity": True,
                "position": False,
                "moments": False,
            },
        },
    ]

    all_measurements = measure_selected(
        label_image=label_image,
        intensity_image=intensity_image,
        measurement_selection=measurement_selection,
    )

    measurement_data = table_to_anndata(
        measurement_table=all_measurements, obs=label_table
    )

    print(measurement_data)

    pca(measurement_data, normalize_data=True)

    # Build the neighbor graph and calculate UMAP
    cluster_features(
        measurement_data,
        method="leiden",
        compute_neighbors=True,
        neighbors_kwargs={"n_pcs": 6},
    )

    umap(measurement_data, compute_neighbors=False)

    measurement_data.write("clustered_shapes.h5ad")

else:
    measurement_data = anndata.read_h5ad("clustered_shapes.h5ad")

sampled_measurements = sample_anndata(
    measurement_data, n_samples_per_group=3, group_by="leiden"
)

viewer = napari.Viewer()
viewer.add_labels(label_image, metadata={"adata": measurement_data})

# widget = QtClusterAnnotatorWidget(viewer)
# viewer.window.add_dock_widget(widget)

napari.run()
