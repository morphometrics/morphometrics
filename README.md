# morphometrics

[![License](https://img.shields.io/pypi/l/morphometrics.svg?color=green)](https://github.com/morphometrics/morphometrics/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/morphometrics.svg?color=green)](https://pypi.org/project/morphometrics)
[![Python Version](https://img.shields.io/pypi/pyversions/morphometrics.svg?color=green)](https://python.org)
[![tests](https://github.com/morphometrics/morphometrics/workflows/tests/badge.svg)](https://github.com/morphometrics/morphometrics/actions)
[![codecov](https://codecov.io/gh/morphometrics/morphometrics/branch/main/graph/badge.svg)](https://codecov.io/gh/morphometrics/morphometrics)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/morphometrics)](https://napari-hub.org/plugins/morphometrics)

A plugin for quantifying shape and neighborhoods from images.

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/plugins/stable/index.html
-->

## Installation

### conda environment file
You can install `morphometrics` via our conda environment file. To do so, first install anaconda or miniconda on
your computer. Then, download the [`environment.yml file`](https://raw.githubusercontent.com/kevinyamauchi/morphometrics/master/environment.yml) (right click the link and "Save as..."). In your terminal,
navigate to the directory you downloaded the `environment.yml` file to:

```bash
cd <path/to/downloaded/environment.yml>
```

Then create the `morphometrics` environment using

```bash
conda env create -f environment.yml
```

Once the environment has been created, you can activate it and use `morphometrics` as described below.

```bash
conda activate morphometrics
```

If you are on Mac OS or Linux install the following:

Mac:

```bash
conda install -c conda-forge ocl_icd_wrapper_apple
```

Linux:

```bash
conda install -c conda-forge ocl-icd-system
```


### Development installation

To install latest development version :

    pip install git+https://github.com/kevinyamauchi/morphometrics.git

## Example applications
<table border="0">
<tr><td>


<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/surface_distance_measurement.gif"
width="300"/>

</td><td>

[measure the distance between surfaces](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/surface_distance_measurement.ipynb)

</td></tr><tr><td>

<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/region_props_plugin.png"
width="300"/>

</td><td>

[napari plugin for measuring properties of segmented objects (regionprops)](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/measure_with_widget.py)

</td></tr><tr><td>

<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/object_classification.png"
width="300"/>

</td><td>

[object classification](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/object_classification.ipynb)

</td></tr><tr><td>

<img src="https://github.com/kevinyamauchi/morphometrics/raw/main/resources/mesh_object.png"
width="300"/>

</td><td>

[mesh binary mask](https://github.com/kevinyamauchi/morphometrics/blob/main/examples/mesh_binary_mask.ipynb)


</td></tr></table>


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"morphometrics" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/kevinyamauchi/morphometrics/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
