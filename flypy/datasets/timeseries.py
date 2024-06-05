#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


import numpy as np
import pandas as pd
import operator as oper


OPERATIONS = {
    "<": oper.lt,
    ">": oper.gt,
    "<=": oper.le,
    ">=": oper.ge,
    "==": oper.eq,
    "!=": oper.ne
}


class TimeSeries:
    """
    Class to organize, filter, and manipulate time series and paired metadata.

    Data is stored in a 2D array while metadata is organized hierarchically in
    a DataFrame.

    Attributes:
        _COL_TRACE (str):
            Name of column in metadata DataFrame that pairs each row of
            metadata to the index of a single trace in the data ndarray.
        _COL_TRIAL (str):
            Name of column in metadata multiindex that separates repeated
            trials of otherwise identical metadata.
        _COL_LABEL (str):
            Name given to column of within-group labels used when computing
            pairwise correlations.
        _COL_GROUP (str):
            Name given to column of correlation group.
        traces (np.ndarray):
            Time series data. Shape = (N traces, N timepoints)
        times (np.ndarray):
            Points along traces attribute time axis. Shape = (N timepoints,).
        meta (pd.DataFrame):
            Hierarchically indexed DataFrame storing metadata associated with
            each trace in data attribute.
    """
    _COL_TRACE = "_trace index"
    _COL_TRIAL = "_trial"
    _COL_LABEL = "_label"
    _COL_GROUP = "_group"

    def __init__(
            self,
            traces: np.ndarray,
            times: np.ndarray,
            trace_names: dict = None,
            metadata: pd.DataFrame = None,
            expand: list = None,
            f_norm: np.ufunc = None,
            agg_trials: list = None
    ):
        """
        Instantiate TimeSeries dataset.

        NOTE: trace_names and metadata args, although both optional, are
        mutually exclusive. trace_names arg is mandatory when creating a new
        instance from scratch, in which case metadata must be None. Metadata
        arg is mandatory when copying a filtered version of an existing
        instance.

        WARNING: When using np.ufunc aggregation in agg_trials arg, ensure that
        len(agg_trials) = 3 and agg_trials[3] = {"axis: 0", ...} such that
        aggregation yields a single trace rather than a scalar. This does not
        apply to custom functions for which func([N traces, N timepoints])
        yields a trace by default.

        Args:
            traces (np.ndarray):
                Array of time series traces for analysis. Must be be at least
                2D with shape = (N traces, ..., N timepoints, ...). If
                traces.ndim > 2, specify expand parameter to unpack axis
                metadata for all extra dimensions.
            times (np.ndarray):
                Array of time point labels shared by all traces in traces arg.
                Must be 1D with shape = (N timepoints,). Arg should reflect
                time axis after normalization if agg_trials arg is provided.
            trace_names (dict, optional):
                Metadata corresponding to first dimension of traces arg.
                Contains the following pairings:
                -   key (scalar):
                        Column name of metadata variable.
                -   item (scalar, iterable):
                        Value(s) of the metadata variable for each trace in
                        traces arg. If scalar, all traces have the same value.
                        Otherwise, must be iterable with shape (N traces,)
                        where item[i] is the label for traces[i].
                Mandatory to create a new TimeSeriesDataset instance.
                Defaults to "None", in which case metadata arg cannot be None.
            metadata (pd.DataFrame, optional):
                Parsed metadata used to create new TimeSeriesDataset instance
                from an existing instance.
                Defaults to "None", in which case metadata is organized via
                trace_names and expand args.
            expand (list, optional):
                Mandatory if traces.ndim > 2, specifies the order in which to
                flatten extra dimensions in traces arg. len(expand) =
                traces.ndim - 2. Each index is a tuple with the following three
                elements:
                -   0 (int):
                        Axis of traces arg to unpack.
                -   1 (str):
                        Name of variable stored in axis.
                    2 (scalar, iterable):
                        Value(s) of metadata variable. As with item in
                        trace_names arg, len(iterable) must equal length of
                        unpacked axis.
                Defaults to "None", in which case traces arg is already 2D.
            f_norm (func, optional):
                Custom function used to normalize each trace individually.
                f_func(trace) = trace.
                Passed to TimeSeries.apply_1d_function.
                Defaults to "None", in which case traces are not normalized.
            agg_trials (list, optional):
                Parameters with which to extract central tendency across
                repeated trials. Contains the following 2 values:
                -   0 (list):
                        Metadata multiindex column names by which to group
                        traces into sets that will be aggregated. All traces
                        with the same set of values across these columns will
                        be aggregated into a single trace.
                -   1 (np.ufunc):
                        Function with which to perform aggregation.
                -   3 (dict, optional):
                        Kwargs to pass to aggregation function.
                Both values passed to TimeSeries.apply_2d_function.
                Defaults to "None", in which case repeated trials are not
                aggregated.

        Examples:
            # test_data shape = (2 traces, 3 time points, 3 recording channels)
            >>> test_data = np.arange(18).reshape(2, 3, 3)
            >>> test_timepoints = np.arange(3)
            >>> test_t_names = {"time": "today", "label": [1, 2]}
            >>> test_expand = [(-1, "channel", ["r", "g", "b"])]
            >>> test_dataset_1 = TimeSeries(
            ... test_data, test_timepoints, test_t_names, expand=test_expand)
            >>> print(test_dataset_1.traces)
            [[ 0  3  6]
             [ 1  4  7]
             [ 2  5  8]
             [ 9 12 15]
             [10 13 16]
             [11 14 17]]
            >>> print(test_dataset_1.meta)
                                        _trace index
            time  label _trial channel
            today 1     0      r                   0
                               g                   1
                               b                   2
                  2     1      r                   3
                               g                   4
                               b                   5

            # test_data shape = (1 trace, 3 time points, 3 recording channels,
            # 2 recording sources)
            # len(test_expand) = test_data.ndim - 2 = 2
            >>> test_data = np.arange(18).reshape(1, 3, 2, 3)
            >>> test_timepoints = np.arange(3)
            >>> test_t_names = {"time": "today", "label": 1}
            >>> test_expand = [(2, "source", ["screen 1", "screen 2"]),
            ... (-1, "channel", ["r", "g", "b"])]
            >>> test_dataset_2 = TimeSeries(
            ... test_data, test_timepoints, test_t_names, expand=test_expand)
            >>> print(test_dataset_2.traces)
            [[ 0  6 12]
             [ 1  7 13]
             [ 2  8 14]
             [ 3  9 15]
             [ 4 10 16]
             [ 5 11 17]]
            >>> print(test_dataset_2.meta)
                                                 _trace index
            time  label _trial source   channel
            today 1     0      screen 1 r                   0
                                        g                   1
                                        b                   2
                               screen 2 r                   3
                                        g                   4
                                        b                   5
        """
        # set instance attributes
        self.traces = np.atleast_2d(traces).copy()
        self.times = times

        # create metadata DataFrame from raw traces and input args
        if trace_names is not None:
            indices = np.arange(traces.shape[0])
            self.meta = pd.DataFrame(data=trace_names, index=indices)
            self._set_index_column(self._COL_TRIAL)
            self._unpack_dims(expand if expand is not None else {})
            self._set_index_column(self._COL_TRACE)
            self.add_index(self.meta.columns.to_list()[:-1], append=False)

        # reset trace column labels to match filtered input data
        elif metadata is not None:
            self.meta = metadata.copy()
            self._set_index_column(self._COL_TRACE)

        else:
            assert ValueError

        norm = self if f_norm is None else self.apply_1d_function(f_norm)
        norm = self if agg_trials is None else self.apply_2d_function(
            cols=agg_trials[0], func=agg_trials[1], **(agg_trials + [{}])[2])
        self.traces = norm.traces
        self.meta = norm.meta
        self.times = norm.times

    def _unpack_dims(
            self,
            expand: list,
    ):
        """
        Unpack extra dimensions in instance traces attribute. Extra dimensions
        are unpacked in the same order as in the input list.

        Args:
            expand (list):
                Refer to TimeSeries.__init__() docstring.
        """
        old_dims = list(range(self.traces.ndim))
        new_dims = list(range(self.traces.ndim))
        for i, (d, key, values) in enumerate(expand):
            # move dimension d to position specified by input order i
            new_dims.remove(old_dims[d])
            new_dims.insert(i + 1, old_dims[d])

            # unpack dimension metadata into long format
            self.meta = self.meta.assign(
                **{key: [values] * self.meta.shape[0]})
            self.meta = self.meta.explode(key, ignore_index=True)

        # reorder and reduce dimensions to yield a 2D array
        self.traces = self.traces.transpose(new_dims)
        self.traces = np.reshape(self.traces, (-1, self.traces.shape[-1]))

    def _set_index_column(
            self,
            col: str
    ):
        """
        Add a column of indices to instance meta attribute. If performed prior
        to TimeSeries._unpack_dims, resulting column will identify unique trial
        in dataset. If performed after TimeSeries._unpack_dims, resulting
        column will identify index of corresponding trace in instance traces
        attribute.

        Args:
            col (str):
                Name of column that contains added index.
        """
        self.meta = self.meta.assign(**{col: list(range(self.meta.shape[0]))})

    def __len__(
            self
    ):
        """
        Returns:
            (int):
                Number of (metadata, time series trace) pairs stored.
        """
        return self.meta.shape[0]

    def __add__(
            self,
            other
    ):
        """
        Concatenate two compatible TimeSeriesDataset instances. Both should
        have compatible hierarchical indices and an equivalent timepoints
        attribute.

        Args:
            other (TimeSeries):
                Compatible instance to add.

        Returns:
            (TimeSeries):
                New instance with pooled metadata and traces.
        """
        assert type(other) == TimeSeries

        cols_self = list(self.meta.index.names)
        cols_other = list(other.meta.index.names)
        cols_total = [c for c in cols_self if c in cols_other]
        metas = [self.meta.copy(), other.meta.copy()]
        for i, m in enumerate(metas):
            m.index = m.index.droplevel(
                [c for c in list(m.index.names) if c not in cols_total])
            m = m.reorder_levels(cols_total)
            metas[i] = m

        data = np.concatenate((self.traces, other.traces), axis=0)
        meta = pd.concat(metas, axis=0)
        return TimeSeries(data, self.times, metadata=meta)

    def __getitem__(
            self,
            col_dict: dict
    ):
        """
        Filter all traces in instance that match a given metadata pattern.

        Args:
            col_dict (dict):
                Metadata pattern by which to filter instance. Contains the
                following pairings:
                -   key (str):
                        Metadata multiindex column name.
                -   item (list):
                        Values along selected metadata axis to include.

        Returns:
            (TimeSeries):
                New instance where all traces fit the metadata pattern in
                key_dict arg.

        Example:
            >>> test_data = np.arange(18).reshape(2, 3, 3)
            >>> test_timepoints = np.arange(3)
            >>> test_t_names = {"time": "today", "label": [1, 2]}
            >>> test_expand = [(-1, "channel", ["r", "g", "b"])]
            >>> test_dataset_1 = TimeSeries(
            ... test_data, test_timepoints, test_t_names, expand=test_expand)
            >>> test_dataset_2 = test_dataset_1[
            ... {"time": ["today"], "label": [1]}]
            >>> print(test_dataset_1.meta)
                                        _trace index
            time  label _trial channel
            today 1     0      r                   0
                               g                   1
                               b                   2
                  2     1      r                   3
                               g                   4
                               b                   5
            >>> print(test_dataset_2.meta)
                                        _trace index
            time  label _trial channel
            today 1     0      r                   0
                               g                   1
                               b                   2
        """
        # filter metadata according to key_dict
        sub_metadata = self.meta
        for key, value in col_dict.items():
            value = (value if isinstance(value, list) else [value])
            sub_metadata = sub_metadata[
                sub_metadata.index.get_level_values(key).isin(value)].copy()

        # filter traces according to filtered metadata
        trace_indices = sub_metadata[self._COL_TRACE].to_list()
        sub_data = np.atleast_2d(self.traces[trace_indices])
        return TimeSeries(sub_data, self.times, metadata=sub_metadata)

    def filter(
            self,
            col: str,
            operation: str,
            threshold
    ):
        """
        Filter instance with binary operation on metadata multiindex level.
        Boilerplate code of __getitem__.

        Args:
            col (str):
                Column name on which to perform filtering.
            operation (str):
                Binary operator with which to threshold column. Valid entries
                are "<", ">", "<=", ">=", "==", "!=".
            threshold (scalar):
                Threshold against which to compare level values

        Returns:
            (TimeSeries):
                New instance filter to only those data for which meta[col]
                operation threshold evaluates True.
        """
        values = self.meta.index.get_level_values(col).unique()
        values = values.to_numpy(copy=True)
        values = values[OPERATIONS[operation](values, threshold)].tolist()
        return self[{col: values}]

    def clip_traces(
            self,
            indices: np.ndarray
    ):
        """
        Extract a subset of timepoints from instances traces attribute.

        Args:
            indices (np.ndarray):
                1D array of index labels to extract.

        Returns:
            (TimeSeries):
                New instance with traces attribute clipped to
                self.traces[:, indices]
        """
        return TimeSeries(
            self.traces[..., indices], self.times[indices], metadata=self.meta)

    def add_index(
            self,
            cols: list,
            append: bool
    ):
        """
        Convert column(s) in instance meta attribute to hierarchical index.

        NOTE: Any entries in cols arg currently in instance meta attribute
        multiindex will be dropped before adding new index level.

        Args:
            cols (list):
                Column name(s) to convert to multiindex.
            append (bool):
                True if new index should be appended to existing multiindex.

        Returns:
            self (TimeSeries): Includes adjusted multiindex.
        """
        for col in cols:
            self.meta.index = (
                self.meta.index.droplevel(col) if col in self.meta.index.names
                else self.meta.index)

        self.meta = self.meta.set_index(cols, append=append)
        return self

    def remove_index(
            self,
            cols: list
    ):
        """
        Convert level(s) of hierarchical index to columns in instance meta
        attribute.

        Args:
            cols (list):
                Metadata multiindex column names to convert.

        Returns:
            self (TimeSeries):
                Includes adjusted multiindex.
        """
        self.meta = self.meta.reset_index(level=cols)
        return self

    def get_index_levels(
            self,
            col
    ):
        """
        Get values of multiindex column as an array.

        Args:
            col (str):
                Metadata column to extract.

        Returns:
            (np.ndarray):
                Metadata column.
        """
        return self.meta.index.get_level_values(col).to_numpy()

    def get_vector_form(
            self,
            cols_source: list,
            col_feature: str,
    ):
        """
        Convert features extracted from instance traces attribute into vectors.
        Call after feature extraction with TimeSeries.apply_scalar_function.

        Args:
            cols_source (list):
                Metadata multiindex column name(s) by which to group feature
                observations into vectors. Act as corresponding labels for
                feature vectors.
            col_feature (str):
                Column name of the desired feature.

        Returns:
            vectors (np.ndarray):
                Array of feature vectors with shape = (N samples, N features).
            labels (np.ndarray):
                Array of labels with shape = (N samples, len(label_keys)).
        """
        _safe = self._COL_TRIAL in self.meta.index.names
        n_levels = len(cols_source)
        levels = cols_source + [self._COL_TRIAL] if _safe else cols_source
        data = self.meta.copy().sort_index().groupby(
            level=levels)[col_feature].agg(lambda x: list(x)).reset_index()
        vectors = np.array(data[col_feature].to_list(), dtype=object)
        labels = data[levels[:n_levels]].to_numpy(copy=True)
        return vectors, labels

    def get_subgroup_counts(
            self,
            cols: list = None
    ):
        """
        Count the number of traces in each unique group of metadata values.

        Args:
            cols (list):
                Metadata multiindex column names by which to group traces for
                counts.

        Returns:
            (pd.Dataframe):
                Filtered instance meta attribute where each row is a unique
                combination of relevant metadata variables and a single column
                reports the number of traces with the same metadata
                combination.
        """
        cols = self.meta.index.names if cols is None else cols
        return self.meta.groupby(level=cols).size()

    def apply_scalar_function(
            self,
            col: str,
            func,
            *args,
            **kwargs
    ):
        """
        Apply function to each trace and populate new metadata column with the
        output. Function should accept a 1D input trace and output a scalar.

        Args:
            col (str):
                Variable name of the scalar output.
            func (function):
                trace_in.shape = (N timepoints,)
                func(trace_in, *args, *kwargs) = scalar.
            *args:
                Arguments, passed to func arg.
            **kwargs:
                Keyword arguments, passed to func arg.

        Returns:
            self (TimeSeries):
                Includes "col arg: func arg" scalar column.

        Example:
            >>> test_data = np.arange(9).reshape(3, 3)
            >>> test_timepoints = np.arange(3)
            >>> test_t_names = {"time": "today", "label": ["a", "b", "a"]}
            >>> test_dataset = TimeSeries(test_data, test_timepoints,
            ... test_t_names).apply_scalar_function(
            ... "mean", np.mean)
            >>> print(test_dataset.traces)
            [[0 1 2]
             [3 4 5]
             [6 7 8]]
            >>> print(test_dataset.meta)
                                _trace index  mean
            time  label _trial
            today a     0                  0   1.0
                  b     1                  1   4.0
                  a     2                  2   7.0
        """
        self.meta[col] = list(
            np.apply_along_axis(func, -1, self.traces, *args, **kwargs))
        return self

    def apply_1d_function(
            self,
            func,
            times: np.ndarray = None,
            *args,
            **kwargs
    ):
        """
        Apply function to each trace and create new class instance with the
        output. Function should accept a 1D input trace and return a 1D output
        trace.

        Args:
            func (function):
                trace_in.shape = (N timepoints,)
                func(trace_in, *args, *kwargs) = trace_out.
            times (np.ndarray):
                Update instance times attribute in case where func arg alters
                trace length.
                Defaults to "None", in which case times attribute is clipped to
                trace_out.size.
            *args:
                Arguments, passed to func arg.
            **kwargs:
                Keyword arguments, passed to func arg.

        Returns:
            (TimeSeries):
                New instance with traces attribute modified by func arg.

        Example:
            >>> test_data = np.arange(9).reshape(3, 3)
            >>> test_timepoints = np.arange(3)
            >>> test_t_names = {"time": "today", "label": ["a", "b", "a"]}
            >>> test_dataset_1 = TimeSeries(
            ... test_data, test_timepoints, test_t_names)
            >>> test_dataset_2 = test_dataset_1.apply_1d_function(np.square)
            >>> print(test_dataset_1.traces)
            [[0 1 2]
             [3 4 5]
             [6 7 8]]
            >>> print(test_dataset_2.traces)
            [[ 0  1  4]
             [ 9 16 25]
             [36 49 64]]
        """
        times = self.times if times is None else times
        traces = np.apply_along_axis(func, -1, self.traces, *args, **kwargs)
        return TimeSeries(traces, times[:traces.shape[-1]], metadata=self.meta)

    def apply_2d_function(
            self,
            cols,
            func,
            times: np.ndarray = None,
            *args,
            **kwargs
    ):
        """
        Apply function to traces aggregated by metadata and create new class
        instance with the output. Function should accept a 2D array of traces
        with dimensions of (N traces, N timepoints) and return a 1D output
        trace.

        Args:
            cols (list):
                Metadata multiindex column names by which to group traces.
                Traces with the same set of metadata labels will be aggregated
                together by func arg.
            func (function):
                traces_in.shape = (N samples_in_group, N timepoints)
                func(traces_in, *args, *kwargs) = trace_out.
            times (np.ndarray):
                Update instance times attribute in case where func arg alters
                trace length.
                Defaults to "None", in which case times attribute is clipped to
                trace_out.shape[1].
            *args:
                Arguments, passed to func arg.
            **kwargs:
                Keyword arguments, passed to func arg.

        Returns:
            (TimeSeries):
                New instance with traces attribute modified by func arg after
                grouping by cols arg.

        Example:
            >>> test_data = np.arange(18).reshape(2, 3, 3)
            >>> test_timepoints = np.arange(3)
            >>> test_t_names = {"time": "today", "label": [1, 2]}
            >>> test_expand = [(-1, "channel", ["r", "g", "b"])]
            >>> test_dataset_1 = TimeSeries(
            ... test_data, test_timepoints, test_t_names, expand=test_expand)
            >>> test_dataset_2 = test_dataset_1.apply_2d_function(
            ... ["time", "label"], np.sum, axis=0)
            >>> print(test_dataset_1.traces)
            [[ 0  3  6]
             [ 1  4  7]
             [ 2  5  8]
             [ 9 12 15]
             [10 13 16]
             [11 14 17]]
            >>> print(test_dataset_1.meta)
                                        _trace index
            time  label _trial channel
            today 1     0      r                   0
                               g                   1
                               b                   2
                  2     1      r                   3
                               g                   4
                               b                   5
            >>> print(test_dataset_2.traces)
            [[ 3 12 21]
             [30 39 48]]
            >>> print(test_dataset_2.meta)
                         _trace index
            time  label
            today 1                 0
                  2                 1
        """
        times = self.times if times is None else times

        # group traces by key and get corresponding indices as list
        groups = self.meta.groupby(level=cols)
        metadata = groups[[self._COL_TRACE]].agg(lambda x: list(x))
        trace_idx = metadata[self._COL_TRACE].to_list()

        # apply func on 2D sub arrays of data attribute
        traces = np.stack(
            [func(self.traces[i], *args, **kwargs) for i in trace_idx], axis=0)
        return TimeSeries(
            traces, times[:traces.shape[1]], metadata=groups.sum())

    def pairwise_corr_against_reference(
            self,
            other=None,
            col_label: str = None,
            col_group: str = None,
            cols_source: list = None,
            label_fill: int = -1
    ):
        """
        Compute the pairwise correlation between traces in instance.

        NOTE: label_fill arg should not be "None" as this value is used to mask
        self comparison (perfect correlation between a trace and itself).

        Args:
            other (TimeSeries, optional):
                Another instance against which to compute correlations.
                Defaults to "None", in which case correlations are computed
                against self and reciprocal correlation values are 1.
            col_label (str, optional):
                Multiindex column name to use as label for each trace prior to
                computing correlation matrix. Must be present in both self and
                "other" arg, if provided.
                Defaults to "None", in which case all traces are treated as if
                they have unique labels.
            col_group (str, optional):
                Multiindex column name by which to group traces into subsets
                for comparison against "other" arg. If "other" is "None", each
                group is compared against itself.
                Defaults to "None", in which case all traces are treated as
                part of the same group.
            cols_source: (list, optional):
                Multiindex column name(s) by which to identify valid
                comparisons for correlation. Only traces with the same set of
                metadata values in these columns will be compared against each
                other. For example, col_source = ["electrode"] indicates that
                traces from a given electrode will only ever be compared to
                other traces from that same electrode.
                Defaults to "None", in which case all traces in each group are
                compared against each other.
            label_fill (int, optional):
                Used as label for correlation values between labels. Must not
                be equal to real labels in either TimeSeries arg.
                Defaults to "-1".

        Returns:
            corr (pd.DataFrame)
                corr.shape = (N groups, N labels).
                corr.loc[g, l] = [within-label correlations in group g]
                corr.loc[g, -1] = [between-label correlations in group g]

        """
        def _corr_by_label(
                _traces_0: np.ndarray,
                _labels_0: np.ndarray,
                _traces_1: np.ndarray,
                _labels_1: np.ndarray,
                _fill: int
        ):
            """
            Nested func. Compute correlation between every combination of
            traces in input arrays. Return list of correlations within each
            label and list of correlations between all labels.

            Args:
                _traces_0 (np.ndarray):
                    _traces_0.shape = (N traces_0, N timepoints).
                _labels_0 (np.ndarray):
                    _labels_0.shape = (N traces_0,).
                _traces_1 (np.ndarray):
                    _traces_1.shape = (N traces_1, N timepoints).
                _labels_1 (np.ndarray):
                    _labels_1.shape = (N traces_1,).

            Returns:
                corr (pd.Series):
                    corr.loc[l] = [correlations within label l].
                    corr.loc[-1] = [correlations between labels].
            """
            mask = np.where(
                _labels_0[:, None] == _labels_1, _labels_1[None, :], _fill)
            mask = mask.astype(object)
            if np.array_equal(_traces_0, _traces_1):
                np.fill_diagonal(mask, None)

            unique = np.unique(_labels_0).tolist() + [_fill]
            corr = np.corrcoef(
                _traces_0, _traces_1)[:_labels_0.size, -_labels_1.size:]
            corr = pd.Series([corr[mask == x] for x in unique], index=unique)
            return corr

        # add col_label and cols_group columns if absent from metadata
        other = self if other is None else other
        cols_source = [] if cols_source is None else cols_source
        meta_self = self.meta.copy()
        meta_other = other.meta.copy()
        if col_label is None:
            col_label = self._COL_LABEL
            meta_self[col_label] = np.arange(meta_self.shape[0])
            meta_other[col_label] = np.arange(meta_self.shape[0])
        else:
            meta_self = meta_self.reset_index(col_label, drop=False)
            meta_other = meta_other.reset_index(col_label, drop=False)

        if col_group is None:
            col_group = self._COL_GROUP
            meta_self[col_group] = 0
            meta_self = meta_self.set_index(col_group, append=True)
        else:
            meta_other.index = meta_other.index.droplevel(col_group)

        # add static comparison group "other" for each unique group in "self"
        groups = meta_self.index.unique(level=col_group).to_list()
        meta_other = meta_other.assign(
            **{col_group: [groups] * meta_other.shape[0]})
        meta_other = meta_other.explode(col_group).set_index(
            col_group, append=True)

        # join self and other metadata, indices within rows will be compared
        _label_other = f"{col_label} other"
        _trace_other = f"{self._COL_TRACE} other"
        meta_self = meta_self.groupby(level=[col_group] + cols_source)[
            [col_label, self._COL_TRACE]].agg(lambda x: list(x))
        meta_other = meta_other.groupby(level=[col_group] + cols_source)[
            [col_label, self._COL_TRACE]].agg(lambda x: list(x)).rename(
            columns={col_label: _label_other, self._COL_TRACE: _trace_other})
        meta_tot = pd.concat(
            [meta_self, meta_other], axis=1, ignore_index=False, join="inner")

        # compute correlations and return output
        meta_tot = meta_tot.apply(
            lambda x: _corr_by_label(
                _traces_0=self.traces[x[self._COL_TRACE]],
                _labels_0=np.array(x[col_label]),
                _traces_1=other.traces[x[_trace_other]],
                _labels_1=np.array(x[_label_other]),
                _fill=label_fill), axis=1)
        return meta_tot
