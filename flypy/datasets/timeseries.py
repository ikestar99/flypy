#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


import numpy as np
import pandas as pd

from itertools import product
from scipy.signal import decimate

from flypy.datasets.coredataset import CoreDataset
from flypy.datasets.featurevector import FeatureVector


class TimeSeries(CoreDataset):
    """
    Class to organize, filter, and manipulate time series and paired metadata.
    Inherits from CoreDataset class.

    ---------------------------------------------------------------------------
    Attributes:
        _TRIAL (str):
            Internal metadata level to distinguish between repeated trials.
        _INDEX (str):
            Internal metadata level to track corresponding index of each datum.
        _OTHER (str):
            Internal suffix used for comparisons against a reference.
        data (np.ndarray):
            Stored data.
        meta (pd.DataFrame):
            Hierarchically indexed DataFrame storing metadata.
        axes (int):
            Number of axes per datum.
        times (np.ndarray):
            Points along traces attribute time axis.

    ---------------------------------------------------------------------------
    Examples:
        Instantiate using "expand" arg.
        >>> # test_data shape = (3 * 3 traces, 5 timepoints)
        >>> test_data = np.arange(45).reshape(3, 3, 5)
        >>> # test_meta shape = (3 * 3 traces, 2 metadata variables)
        >>> test_meta = pd.DataFrame(
        ... {"day": ["Mon", "Mon", "Tues"], "session": ["AM", "PM", "PM"]})
        >>> # test_times.shape = (5 timepoints,)
        >>> test_times = np.arange(5)
        >>> test_expand = [(-2, "electrode", [0, 1, 2])]
        >>> test_dataset = TimeSeries(
        ... test_data, test_meta, test_times, test_expand)
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29],
               [30, 31, 32, 33, 34],
               [35, 36, 37, 38, 39],
               [40, 41, 42, 43, 44]])
        >>> test_dataset.meta
                                                _data_index
        day  session _repeated_trial electrode
        Mon  AM      0               0                    0
                                     1                    1
                                     2                    2
             PM      1               0                    3
                                     1                    4
                                     2                    5
        Tues PM      2               0                    6
                                     1                    7
                                     2                    8

        -----------------------------------------------------------------------
        Subsample every other timepoint.
        Note that this operation does not alter instance metadata.
        >>> test_idx = np.arange(0, test_dataset.data.shape[-1], 2)
        >>> test_sub = test_dataset.subsample(test_idx)
        >>> test_sub.times
        array([0, 2, 4])
        >>> test_sub.data
        array([[ 0,  2,  4],
               [ 5,  7,  9],
               [10, 12, 14],
               [15, 17, 19],
               [20, 22, 24],
               [25, 27, 29],
               [30, 32, 34],
               [35, 37, 39],
               [40, 42, 44]])

        -----------------------------------------------------------------------
        Downsample traces by a factor of two.
        Note that this operation does not alter instance metadata.
        >>> test_factor = 2
        >>> test_down = test_dataset.downsample(test_factor)
        >>> test_down.times
        array([0, 2])
        >>> np.around(test_down.data, 3)
        array([[ 0.   ,  2.021],
               [ 4.943,  6.964],
               [ 9.886, 11.906],
               [14.829, 16.849],
               [19.771, 21.792],
               [24.714, 26.735],
               [29.657, 31.677],
               [34.6  , 36.62 ],
               [39.542, 41.563]])

        -----------------------------------------------------------------------
        Extract mean of each trace and collapse metadata level into vector.
        Note that feature vector extraction creates a unique vector for each
        repeated trial. In this case, each vector represents:
        <mean(electrode 0), mean(electrode 1), mean(electode 2)>
        for every remaining set of metadata level values.
        >>> test_vector = test_dataset.to_feature_vector(
        ... np.mean, ["electrode"])
        >>> test_vector.data
        array([[ 2.,  7., 12.],
               [17., 22., 27.],
               [32., 37., 42.]])
        >>> test_vector.meta
                                      _data_index
        day  session _repeated_trial
        Mon  AM      0                          0
             PM      1                          1
        Tues PM      2                          2

        -----------------------------------------------------------------------
        Extract mean of each trace and collapse metadata level into vector.
        Note that the "AM" session is dropped entirely as vectors from day
        "Tues" lack this feature. Therefore, each vector represents:
        <mean(PM electrode 0), mean(PM electrode 1), mean(PM electrode 2)>
        >>> test_vector = test_dataset.to_feature_vector(
        ... np.mean, ["session", "electrode"])
        >>> test_vector.data
        array([[17., 22., 27.],
               [32., 37., 42.]])
        >>> test_vector.meta
                              _data_index
        day  _repeated_trial
        Mon  1                          0
        Tues 2                          1

        -----------------------------------------------------------------------
        Extract individual timepoints as vectors.
        Note that the "AM" session is dropped entirely as vectors from day
        "Tues" lack this feature. Therefore, each vector represents:
        <PM electrode 0 time n, PM electrode 1 time n, PM electrode 2 time n>
        >>> test_vector = test_dataset.to_time_vector(
        ... ["session", "electrode"], "time_in_s")
        >>> test_vector.data
        array([[15, 20, 25],
               [16, 21, 26],
               [17, 22, 27],
               [18, 23, 28],
               [19, 24, 29],
               [30, 35, 40],
               [31, 36, 41],
               [32, 37, 42],
               [33, 38, 43],
               [34, 39, 44]])
        >>> test_vector.meta
                                        _data_index
        day  _repeated_trial time_in_s
        Mon  1               0                    0
                             1                    1
                             2                    2
                             3                    3
                             4                    4
        Tues 2               0                    5
                             1                    6
                             2                    7
                             3                    8
                             4                    9
    """
    def __init__(
            self,
            data: np.ndarray,
            meta: pd.DataFrame,
            times: np.ndarray = None,
            expand: list = None,
    ):
        """
        Instantiate TimeSeries instance.
        See CoreDataset.__init__ docstring for more detailed description.

        NOTE: data.ndim = 1 (N traces) + 1 (1d trace) + len(expand).

        Args:
            data (np.ndarray):
                Array of time series traces. Must be at least 2d.
                data.shape = (N traces, ..., N timepoints, ...).
            meta (pd.DataFrame):
                Metadata corresponding to first axis of data arg.
                meta.iloc[i, :] = set of labels for data[i].
            times (np.ndarray, optional):
                Labels along time axis. Must be 1d, shape = (N timepoints,).
                Defaults to np.arange(trace.size)
            expand (list, optional):
                Mandatory if data.ndim > 2. len(expand) = traces.ndim - 2.
                See CoreDataset.__init__ docstring
        """
        # set instance attributes
        super(TimeSeries, self).__init__(data, meta, axes=1, expand=expand)
        self.times = np.arange(self.data.shape[-1]) if times is None else times

    def subsample(
            self,
            idx
    ):
        """
        Subsample all traces stored in instance.

        Args:
            idx (slice, array-like):
                Indices to extract along time axis of instance data.
                sub_trace = raw_trace[idx]

        Returns:
            (TimeSeries):
                New instance with subsampled data.

        """
        return TimeSeries(self.data[..., idx], self.meta, self.times[idx])

    def downsample(
            self,
            factor: int
    ):
        """
        Downsample all traces stored in instance.

        Args:
            factor (int):
                Downsampling factor.
                len(down_trace) = len(raw_trace) // factor

        Returns:
            (TimeSeries):
                New instance with downsampled data.
        """
        def _safe_downsample(x, q):
            pad = max(0, 28 - x.size)
            end = x.size // q
            return decimate(np.pad(x, (0, pad), mode="edge"), q=q)[:end]

        down = self.apply_function(_safe_downsample, q=factor)
        down.times = _safe_downsample(
            down.times, factor).astype(down.times.dtype)
        return down

    def to_feature_vector(
            self,
            func,
            levels: list,
            *args,
            **kwargs
    ):
        """
        Convert features extracted from instance traces attribute into vectors.

        NOTE: Function filters by each level to be collapsed such that all
        vectors have the same set and ordering of features.

        Args:
            func (function):
                Function with which to extract scalar from each trace
            levels (list):
                Metadata level names to collapse into individual feature
                vectors.
            *args:
                Arguments. Passed to func arg.
            **kwargs:
                Keyword arguments. Passed to func arg.

        Returns:
            (FeatureVector):
                Instance with feature-extracted vectors.
        """
        _temp = "_temporary"

        # extract feature from each trace
        sub_meta = self.extract_statistic(func, _temp, *args, **kwargs)

        # ensure all vectors have the same features
        for n in levels:
            # find common values across each metadata level to collapse
            common_value = set(sub_meta.index.get_level_values(n))
            other_levels = [
                i for i in sub_meta.index.names if i not in (n, self._TRIAL)]
            common_value = common_value.intersection(*[
                set(g.index.get_level_values(n))
                for _, g in sub_meta.groupby(level=other_levels)])

            # filter features to common values
            sub_meta = sub_meta[
                sub_meta.index.get_level_values(n).isin(list(common_value))]

        # group vectors by metadata variables other than levels arg
        groups = [i for i in list(self.meta.index.names) if i not in levels]
        sub_meta = sub_meta.groupby(level=groups).agg(lambda x: list(x))

        # convert features into vectors by aggregating across input levels
        data_vec = np.array(sub_meta[_temp].to_list())
        return FeatureVector(data_vec, sub_meta[[self._INDEX]])

    def to_time_vector(
            self,
            levels: list,
            col_time: str
    ):
        """
        Convert individual time points into feature vectors.

        NOTE: Function filters by each level to be collapsed such that all
        vectors have the same set and ordering of features.

        Args:
            levels (list):
                Metadata level names to collapse into individual feature
                vectors per time unit.
            col_time (str):
                Name of time metadata variable.

        Returns:
            vectors (FeatureVector):
                Instance with time-extracted vectors.
        """
        sub_meta = self.meta.copy()

        # ensure all vectors have the same features
        for n in levels:
            # find common values across each metadata level to collapse
            common_value = set(sub_meta.index.get_level_values(n))
            other_levels = [
                i for i in sub_meta.index.names if i not in (n, self._TRIAL)]
            common_value = common_value.intersection(*[
                set(g.index.get_level_values(n))
                for _, g in sub_meta.groupby(level=other_levels)])

            # filter features to common values
            sub_meta = sub_meta[
                sub_meta.index.get_level_values(n).isin(list(common_value))]

        # group vectors by metadata variables other than levels arg
        groups = [i for i in list(self.meta.index.names) if i not in levels]
        sub_meta = sub_meta.groupby(level=groups)[[self._INDEX]].agg(
            lambda x: list(x))

        # extract vectors by transposing traces to (N timepoints, N trials)
        sub_data = np.concatenate([
            self.data[idx].transpose()
            for idx in sub_meta[self._INDEX].tolist()], axis=0)

        # create a new index level specifying the recording time of each vector
        sub_meta = sub_meta.assign(
            **{col_time: [self.times] * sub_meta.shape[0]}).explode(col_time)
        sub_meta = sub_meta.set_index(col_time, append=True)
        return FeatureVector(sub_data, sub_meta)

    def cont_cross_corr(
            self,
            cont,
            label: str,
            group: str,
            ignore: list = None,
    ):
        """
        Compute the pairwise correlation between traces in instance.

        NOTE: label_fill arg should not be "None" as this value is used to mask
        self comparison (perfect correlation between a trace and itself).

        Args:
            cont (TimeSeries):
                Static instance against which to compute correlations.
            label (str):
                Metadata level name to use as label for each trace prior to
                computing correlation matrix. Must be present in both "self"
                and "cont".
            group (str):
                Metadata level name by which to group traces into subsets for
                comparison against "cont".
            ignore (list, optional):
                Metadata level names to ignore when identifying valid
                comparisons.
                Defaults to "None", in which case comparisons are made within
                all metadata levels outside of those specified by "label" and
                "group" args.

        Returns:
            (pd.DataFrame)
                meta_corr.shape = (N groups, N labels).
                meta_corr.loc[g, l] = [within-label correlations in group g]
        """
        def _corr_by_label(
                _t_s: np.ndarray,
                _l_s: np.ndarray,
                _t_c: np.ndarray,
                _l_c: np.ndarray,
        ):
            """
            Nested func. Compute max cross correlation between two sets of
            traces, within labels.

            Args:
                _t_s (np.ndarray):
                    _traces_self.shape = (N self traces, N timepoints).
                _l_s (np.ndarray):
                    _labels_self.shape = (N self traces,).
                _t_c (np.ndarray):
                    _traces_control.shape = (N control traces, N timepoints).
                _l_c (np.ndarray):
                    _labels_control.shape = (N control labels,).

            Returns:
                (pd.Series):
                    .loc[l] = [max cross correlations within label l].
            """
            unique = np.unique(_l_s).tolist()
            cross_corr = [
                [
                    np.max(
                        np.correlate((s - np.mean(s)) / np.std(s),
                                     (r - np.mean(r)) / np.std(r), mode="full")
                        / (len(s)))
                    for s, r in product(_t_s[_l_s == i], _t_c[_l_c == i])]
                for i in unique]
            return pd.Series(cross_corr, index=unique)

        # compute and return correlations
        return self.grouped_control(cont, _corr_by_label, label, group, ignore)
