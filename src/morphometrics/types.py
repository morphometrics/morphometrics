import numpy as np
import pandas as pd

# pixel values are signal magnitude
IntensityImage = np.ndarray

# pixel values are a calculated feature value
FeatureImage = np.ndarray

# pixel values are bool
BinaryImage = np.ndarray

# pixel values are class/instance membership
LabelImage = np.ndarray

# must index must be the label value
LabelMeasurementTable = pd.DataFrame
