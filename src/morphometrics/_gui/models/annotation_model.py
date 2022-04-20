from enum import Enum
from typing import List, Optional, Union

import anndata
import napari
import numpy as np
import pandas as pd
from napari.utils.events import EmitterGroup, Event

from ...explore.sample import sample_anndata, sample_pandas


class TableSource(Enum):
    ANNDATA = "anndata"
    LAYER_FEATURES = "layer_features"


class ClusterAnnotationModel:
    def __init__(self):
        self.events = EmitterGroup(
            source=self,
            annotation_classes=Event,
            annotating=Event,
            auto_advance=Event,
            layer=Event,
            sample_annotated=Event,
            selected_sample=Event,
        )

        self._layer = None
        self._table_source = TableSource.LAYER_FEATURES
        self._label_column = "label_value"
        self._sample_data = None
        self._annotating = False
        self._group_by = None
        self._n_samples_per_group = 10
        self._random_seed = 42
        self._auto_advance = True

        # index of the label in the label image for the selected observation
        self._selected_sample = 0

    @property
    def layer(self) -> Optional[napari.layers.Labels]:
        return self._layer

    @layer.setter
    def layer(self, layer: Optional[napari.layers.Layer]):
        self._layer = layer
        self.events.layer()

    @property
    def table_source(self) -> str:
        """The source of the features table from the selected layer.

        If anndata, the source is assumed to be from layer.metadata["adata"].
        If layer_features, the source is assuemd to be from layer.features
        """
        return self._table_source.value

    @table_source.setter
    def table_source(self, table_source: Union[TableSource, str]):
        table_source = TableSource(table_source)
        self._table_source = table_source

    @property
    def feature_table(self) -> Optional[anndata.AnnData]:
        if self._layer is None:
            return None

        if self._table_source == TableSource.ANNDATA:
            return self._layer.metadata["adata"]
        else:
            return self._layer.features

    @property
    def annotations(self) -> pd.Series:
        """The annotations that have been added to the sampled data"""
        if self._table_source == TableSource.ANNDATA:
            return self._sample_data.obs[self._label_column]
        else:
            return self._sample_data[self._label_column]

    @property
    def annotation_table(self) -> pd.DataFrame:
        """The full table to which the annotations are being added.

        If the feature table is AnnData, returns obs.
        If the feature table is a pandas DataFrame, returns the full dataframe.
        """

        if self._table_source == TableSource.ANNDATA:
            return self.feature_table.obs
        else:
            return self.feature_table

    @property
    def sample_annotation_table(self) -> pd.DataFrame:
        """The sample table to which the annotations are being added.

        If the feature table is AnnData, returns obs.
        If the feature table is a pandas DataFrame, returns the full dataframe.
        """

        if self._table_source == TableSource.ANNDATA:
            return self._sample_data.obs
        else:
            return self._sample_data

    @property
    def annotation_classes(self) -> List[Union[str, int]]:
        return self._annotation_classes

    @annotation_classes.setter
    def annotation_classes(self, annotation_classes: List[Union[str, int]]):
        self._annotation_classes = annotation_classes
        self.events.annotation_classes()

    @property
    def annotating(self) -> bool:
        """State variable that is set to True when annotating.

        Setting annotating to True/False sets up/tears down the annotation mode.
        """
        return self._annotating

    @annotating.setter
    def annotating(self, annotating: bool):
        if annotating == self.annotating:
            # if the value is not changing, return
            return
        if self.layer is None:
            self._annotating = False

        if annotating is True:
            self._initialize_annotation()
        else:
            self._teardown_annotation()
        self._annotating = annotating
        self.events.annotating()

    def _initialize_annotation(self):
        """Initialize the samples from the feature table for annotation."""
        if self.feature_table is None:
            return

        if self._label_column not in self.annotation_table:
            self.annotation_table[self._label_column] = np.nan

        if self._table_source == TableSource.ANNDATA:
            self._sample_data = sample_anndata(
                self.feature_table,
                group_by=self._group_by,
                n_samples_per_group=self._n_samples_per_group,
                random_seed=self._random_seed,
            )
        else:
            self._sample_data = sample_pandas(
                self.feature_table,
                group_by=self._group_by,
                n_samples_per_group=self._n_samples_per_group,
                random_seed=self._random_seed,
            )

    def _teardown_annotation(self):
        """After annotation has completed, add the annotations back into the original feature table."""
        self._update_table(
            self.annotation_table, self.sample_annotation_table[self._label_column]
        )
        self._sample_data = None

    @property
    def n_samples(self) -> Optional[int]:
        """The total number of samples.
        Returns 0 if no samples have been initailized.
        """
        if self._sample_data is None:
            return 0
        return len(self._sample_data)

    @property
    def selected_sample(self) -> int:
        return self._selected_sample

    @selected_sample.setter
    def selected_sample(self, selected_sample):
        if (self._layer is None) or (self._sample_data is None):
            return
        self._selected_sample = selected_sample
        selected_sample_row = self.selected_sample_row

        if self._table_source == TableSource.ANNDATA:
            label_value = selected_sample_row.obs["label"].values[0]
        else:
            label_value = selected_sample_row["label"]
        self._layer.selected_label = label_value

        self.events.selected_sample()

    def next_sample(self):
        """Incrememnt the selected sample"""
        if self._sample_data is None:
            # do nothing if the sample data hasn't been set
            return
        self.selected_sample = (self.selected_sample + 1) % self.n_samples

    def previous_sample(self):
        """Decrement the selected sample"""
        if self._sample_data is None:
            # do nothing if the sample data hasn't been set
            return
        self.selected_sample = (
            (self.selected_sample - 1) + self.n_samples
        ) % self.n_samples

    @property
    def selected_sample_row(self) -> Optional[anndata.AnnData]:
        """The AnnData row corresponded to the currently selected observation"""
        if self._sample_data is None:
            return None
        else:
            if self._table_source == TableSource.ANNDATA:
                return self._sample_data[self.selected_sample, :]
            else:
                return self._sample_data.iloc[self.selected_sample, :]

    @property
    def auto_advance(self) -> bool:
        """Flag set to true when the next sample is automatically selected after making an annotation."""
        return self._auto_advance

    @auto_advance.setter
    def auto_advance(self, auto_advance: bool):
        self._auto_advance = auto_advance
        self.events.auto_advance()

    def start_annotation(
        self,
        annotation_classes: List[str] = [
            "true_positive",
            "false_negative",
            "false_positive",
        ],
        group_by: Optional[str] = None,
        n_samples_per_group: int = 10,
        random_seed: int = 42,
    ):
        if self.layer is None:
            return

        if self.annotating is False:
            if not self._validate_n_samples(
                group_by=group_by, n_samples_per_group=n_samples_per_group
            ):
                raise ValueError(
                    "n_samples_per_group is greater than the number of observations"
                )
            # set the labels in the QtLabelSelectWidget
            self.annotation_classes = annotation_classes

            self._group_by = group_by
            self._n_samples_per_group = n_samples_per_group
            self._random_seed = random_seed

            self.annotating = True
            self.selected_sample = 0
        else:
            self.annotating = False

    def _update_table(self, df: pd.DataFrame, new_column: pd.Series):
        column_name = new_column.name
        for index, value in new_column.iteritems():
            df.at[index, column_name] = value

    def _get_group_by_keys(self, combo_widget=None) -> List[str]:
        """Get the valid columns to group the features table by.

        This is generally used by magicgui to determine the valid choices
        for the widget generated from self.start_annotation().

        Returns
        -------
        group_by_keys : List[str]
            A list of valid keys to group the features by.
        """
        if self._layer is None:
            return []

        return self.annotation_table.columns.tolist()

    def _validate_n_samples(
        self, group_by: Optional[str], n_samples_per_group: int
    ) -> bool:
        """Check if the number of samples works with the selected table.

        Note: returns False if self._layer is None.
        """
        if self._layer is None:
            return False

        if group_by is not None:
            max_counts = self.annotation_table[group_by].value_counts().min()
        else:
            max_counts = len(self.annotation_table)
        return n_samples_per_group <= max_counts

    def _annotate_selected_sample(self, annotation_value):
        """Set the currently selected observation to the specified annotation value"""
        self.sample_annotation_table.iat[
            self.selected_sample,
            self.sample_annotation_table.columns.get_loc(self._label_column),
        ] = annotation_value

        self.events.sample_annotated()
        if self.auto_advance is True:
            self.next_sample()
