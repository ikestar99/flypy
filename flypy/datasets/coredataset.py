#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4 13:06:21 2024
@author: ike
"""


import numpy as np
import pandas as pd
import operator


OPERATIONS = {
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne
}


class CoreDataset:
    """
    CoreDataset is a parent class inherited by all other datasets in the
    flypy.datasets module. Fundamentally, a CoreDataset object stores a mapping
    between a set of metadata variables and its corresponding datum, which may
    be a scalar, 1d sequenece, 2d image, or higher dimensional.

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

    ---------------------------------------------------------------------------
    Examples:
        Instantiate with 1d datum.
        >>> # test_data shape = (3 data points, 5 observations per datum)
        >>> test_data = np.arange(15).reshape(3, 5)
        >>> # test_meta shape = (3 data points, 2 metadata variables)
        >>> test_meta = pd.DataFrame(
        ... {"day": "Mon", "label": [1, 1, 2]})
        >>> # test_axes = number of axes in each datum, 1 if datum is 1d array
        >>> test_axes = 1
        >>> test_dataset = CoreDataset(test_data, test_meta, test_axes)
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14]])
        >>> test_dataset.meta
                                   _data_index
        day label _repeated_trial
        Mon 1     0                          0
                  1                          1
            2     2                          2

        -----------------------------------------------------------------------
        Instantiate using "expand" arg.
        Note that an internal _repeated trial" metadata level distinguishes
        between repetitions of the same set of metadata values.
        >>> # test_data shape = (3 * 2 data points, 5 observations per datum)
        >>> test_data = np.arange(30).reshape(3, 2, 5)
        >>> test_expand = [(1, "channel", ["blue", "red"])]
        >>> test_dataset = CoreDataset(
        ... test_data, test_meta, test_axes, test_expand)
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]])
        >>> test_dataset.meta
                                           _data_index
        day label _repeated_trial channel
        Mon 1     0               blue               0
                                  red                1
                  1               blue               2
                                  red                3
            2     2               blue               4
                                  red                5

        -----------------------------------------------------------------------
        Filter instance using __getitem__.
        >>> test_slice = test_dataset[1:6:2]
        >>> test_slice.data
        array([[ 5,  6,  7,  8,  9],
               [15, 16, 17, 18, 19],
               [25, 26, 27, 28, 29]])
        >>> test_slice.meta
                                           _data_index
        day label _repeated_trial channel
        Mon 1     0               red                0
                  1               red                1
            2     2               red                2
        >>> test_slice = test_dataset[{"label": {">": 0}, "channel": ["blue"]}]
        >>> test_slice.data
        array([[ 0,  1,  2,  3,  4],
               [10, 11, 12, 13, 14],
               [20, 21, 22, 23, 24]])
        >>> test_slice.meta
                                           _data_index
        day label _repeated_trial channel
        Mon 1     0               blue               0
                  1               blue               1
            2     2               blue               2

        -----------------------------------------------------------------------
        Add two instances together.
        Note that "label" and "channel" metadata levels are dropped after
        addition since they aren't shared across operands.
        >>> test_data = np.arange(10).reshape(2, 5)
        >>> test_meta_1 = pd.DataFrame({"day": "Mon", "label": [1, 2]})
        >>> test_meta_2 = pd.DataFrame({"day": "Tue", "channel": [5, 6]})
        >>> test_axes = 1
        >>> test_dataset_1 = CoreDataset(test_data, test_meta_1, test_axes)
        >>> test_dataset_2 = CoreDataset(-test_data, test_meta_2, test_axes)
        >>> test_add = test_dataset_1 + test_dataset_2
        >>> test_add.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [ 0, -1, -2, -3, -4],
               [-5, -6, -7, -8, -9]])
        >>> test_add.meta
                             _data_index
        _repeated_trial day
        0               Mon            0
        1               Mon            1
        2               Tue            2
        3               Tue            3

        -----------------------------------------------------------------------
        Extract scalar summary statistic from each datum.
        >>> test_dataset.data
        array([[ 0,  1,  2,  3,  4],
               [ 5,  6,  7,  8,  9],
               [10, 11, 12, 13, 14],
               [15, 16, 17, 18, 19],
               [20, 21, 22, 23, 24],
               [25, 26, 27, 28, 29]])
        >>> test_dataset.extract_statistic(np.mean, "mean")
                                           _data_index  mean
        day label _repeated_trial channel
        Mon 1     0               blue               0   2.0
                                  red                1   7.0
                  1               blue               2  12.0
                                  red                3  17.0
            2     2               blue               4  22.0
                                  red                5  27.0

        -----------------------------------------------------------------------
        Apply function across channels
        >>> test_agg = test_dataset.apply_function(
        ... np.mean, ["label"], axis=0)
        >>> test_agg.data
        array([[10., 11., 12., 13., 14.],
               [15., 16., 17., 18., 19.]])
        >>> test_agg.meta
                                     _data index
        day channel _repeated trial
        Mon blue    0                          0
            red     1                          1
        >>> test_agg = test_dataset.apply_function(
        ... np.median, ["label", "channel"], axis=0)
        >>> test_agg.data
        array([[ 2.5,  3.5,  4.5,  5.5,  6.5],
               [12.5, 13.5, 14.5, 15.5, 16.5],
               [22.5, 23.5, 24.5, 25.5, 26.5]])
        >>> test_agg.meta
                             _data index
        day _repeated trial
        Mon 0                          0
            1                          1
            2                          2
    """
    _TRIAL = "_repeated_trial"
    _INDEX = "_data_index"
    _OTHER = "_other"

    def __init__(
            self,
            data: np.ndarray,
            meta: pd.DataFrame,
            axes: int,
            expand: list = None
    ):
        """
        Instantiate CoreDataset instance.

        NOTE: If "axes" arg >= 1, all data points must have the same shape.
        data.ndim = 1 + axes + len(expand).

        Example:
            -   data.shape = (5, 3, 20, 20)
            -   meta.shape = (5, 4)
            -   axes = 2
            -   expand = [(1, "channel", ["r", "g", "b"])]
            This set of parameters creates a CoreDataset instance with 15 data
            points, where each datum is a 20x20 np.ndarray. Each datum has
            5 corresponding metadata variables, 4 from the columns of meta and
            a 5th from unpacking axis 1 into a "channel" variable where data[0]
            is labeled "r", data[1] is labeled "g", and data[2] is labeled "b".

        Args:
            data (np.ndarray):
                Array of raw data. Can have arbitrary number of axes. First
                axis must correspond to number of data points stored in
                instance.
            meta (pd.DataFrame):
                Metadata corresponding to each data point. Must have a separate
                row for each data point, such that:
                meta.shape[0] = data.shape[0]
                Each column is a separate metadata variable where:
                meta.loc[meta.index[i], var]
                is the "var" metadata value for data[i].
            axes (int):
                Number of axes in each datum. 0 if each datum is a scalar, 1 if
                each datum is a 1d array, 2 if each datum is an image, etc.
            expand (list):
                Instructions with which to unpack extra axes in data arg, if
                present. Each index is a tuple with the following three
                elements:
                -   0 (int):
                        Axis to unpack.
                -   1 (str):
                        Name of variable stored in axis.
                -   2 (scalar, iterable):
                        Value(s) of metadata variable. len(iterable) must equal
                        length of unpacked axis.
                Defaults to "None", in which case no axes are unpacked.
        """
        # remove existing multiindex if present
        if list(meta.index.names)[0] is not None or len(meta.index.names) > 1:
            meta = meta.reset_index(drop=False)

        # add level to distinguish between repeated trials if not present
        if self._TRIAL not in meta.columns:
            meta = meta.assign(**{self._TRIAL: np.arange(meta.shape[0])})

        # unpack extra dimensions if present
        if expand is not None:
            # add level to distinguish between repeated trials
            data, meta = self._unpack_dims(data, meta, axes, expand)

        # first axis of data corresponds to N trials
        while data.ndim < axes + 1:
            data = data[np.newaxis]

        # add column pairing metadata to data index along first axis
        meta = meta.assign(**{self._INDEX: np.arange(meta.shape[0])})

        # create hierarchical index
        levels = [i for i in list(meta.columns) if i != self._INDEX]
        meta = meta.set_index(levels, append=False)

        if data.shape[0] != meta.shape[0]:
            raise ValueError(
                f"With data shape {data.shape} expected {data.shape[0]} " +
                f"metadata rows but got {meta.shape[0]} instead")

        # set instance attributes
        self.data = data
        self.meta = meta.sort_index(inplace=False)
        self.axes = axes

    def __len__(
            self
    ):
        """
        Returns:
            (int):
                Number of (metadata, data) pairs stored.
        """
        return self.meta.shape[0]

    def __add__(
            self,
            other
    ):
        """
        Concatenate two compatible dataset instances. Instances are joined
        along first axis of instance data and meta attributes. Both must have
        identical data attribute shapes along all other aces.

        NOTE: Returned hierarchical index is limited to overlap shared between
        instances.

        Args:
            other (self.__class__):
                Compatible instance to add.

        Returns:
            (self.__class__):
                New instance with pooled metadata and traces.
        """
        assert isinstance(other, self.__class__)

        # find shared metadata levels
        levels = list(set(self.meta.index.names) & set(other.meta.index.names))
        levels = [i for i in list(self.meta.index.names) if i in levels]
        metas = [self.meta.copy(), other.meta.copy()]

        # adjust other trial numbers to avoid overlap with self trial numbers
        if self._TRIAL in levels:
            idx = list(metas[1].index.names).index(self._TRIAL)
            new_min = 1 + metas[0].index.get_level_values(self._TRIAL).max()
            metas[1].index = metas[1].index.set_levels(
                metas[1].index.levels[idx].astype(int) + new_min, level=idx)

        # remove metadata levels unique to each instance
        for i, m in enumerate(metas):
            m.index = m.index.droplevel(
                [i for i in list(m.index.names) if i not in levels])
            metas[i] = m.reorder_levels(levels)

        # concatenate data and metadata
        sub_data = np.concatenate((self.data, other.data), axis=0)
        sub_meta = pd.concat(metas, axis=0)
        return self._safe_copy(sub_data, sub_meta)

    def __getitem__(
            self,
            pattern
    ):
        sub_meta = self.meta.copy()

        # get metadata level values
        if type(pattern) is str:
            return self.meta.index.get_level_values(pattern).to_numpy()

        # filter by numerical index
        if type(pattern) in (int, slice):
            pattern = [pattern] if type(pattern) is int else pattern
            sub_meta = sub_meta.iloc[pattern]

        # filter by boolean mask
        elif type(pattern) in (list, np.ndarray):
            sub_meta = (
                sub_meta.loc[pattern] if type(pattern[0]) is bool
                else sub_meta.iloc[pattern])

        # filter by operation on index level
        elif type(pattern) is dict:
            for key, value in pattern.items():
                # filter by {level: [operation, threshold]}
                if len(value) == 2 and value[0] in OPERATIONS:
                    mask = OPERATIONS[value[0]](self[key], value[1])
                    sub_meta = sub_meta.loc[mask]

                # filter by {level: [values to include]}
                else:
                    sub_meta = sub_meta[
                        sub_meta.index.isin(np.atleast_1d(value), level=key)]

        # filter data attribute to match filtered metadata
        sub_data = self.data[sub_meta[self._INDEX].to_list()].copy()
        return self._safe_copy(sub_data, sub_meta)

    @staticmethod
    def _unpack_dims(
            data: np.ndarray,
            meta: pd.DataFrame,
            axes: int,
            expand: list
    ):
        """
        Unpack extra dimensions in instance traces attribute. Extra dimensions
        are unpacked in the same order as in the input list.

        Args:
            Refer to TimeSeries.__init__() docstring.
        """
        old_dims = list(range(data.ndim))
        new_dims = list(range(data.ndim))
        for i, (d, key, values) in enumerate(expand):
            # move dimension d to position specified by input order i
            new_dims.remove(old_dims[d])
            new_dims.insert(i + 1, old_dims[d])

            # unpack dimension metadata into long format
            meta = meta.assign(**{key: [values] * meta.shape[0]}).explode(
                key, ignore_index=True)

        # reorder and reduce dimensions to yield a 2D array
        data = data.transpose(new_dims)
        data = data.reshape(-1, *data.shape[-axes:])
        return data, meta

    def _safe_copy(
            self,
            sub_data: np.ndarray,
            sub_meta: pd.DataFrame
    ):
        # create a new instance from the same class as self
        kwargs = {"axes": self.axes} if type(self) is CoreDataset else {}
        other = self.__class__(data=sub_data, meta=sub_meta, **kwargs)

        # copy attributes not shared by parent class in new instance
        for attr in self.__dict__:
            if attr not in ("data", "meta"):
                setattr(other, attr, getattr(self, attr))

        return other

    def extract_statistic(
            self,
            func,
            col: str,
            levels: list = None,
            *args,
            **kwargs
    ):
        """
        Apply function to each trace and populate new metadata column with the
        output. Function should output a scalar.

        Args:
            func (function):
                trace_in.shape = (N timepoints,)
                func(trace_in, *args, *kwargs) = scalar.
            col (str):
                Variable name of the scalar output.
            levels (list, optional):
                Metadata level names that should be collapsed. Data indices
                with identical metadata except along axes specified by levels
                arg will be grouped together prior to applying func arg.
                Defaults to "None", in which case no grouping is performed.
            *args:
                Arguments, passed to func arg.
            **kwargs:
                Keyword arguments, passed to func arg.

        Returns:
            (pd.DataFrame):
                Includes "col arg: func arg" scalar column.
        """
        # summarize each datum to a single statistic
        sub_meta = self.meta.sort_index(inplace=False)
        if levels is not None:
            me = list(sub_meta.index.names)
            groups = [i for i in me if i not in levels + [self._TRIAL]]
            sub_meta = sub_meta.groupby(level=groups)[[self._INDEX]].agg(
                lambda x: list(x))

        sub_meta = sub_meta.assign(
            **{col: [
                func(self.data[i], *args, **kwargs)
                for i in sub_meta[self._INDEX].tolist()]})
        return sub_meta

    def apply_function(
            self,
            func,
            levels: list = None,
            *args,
            **kwargs
    ):
        """
        Apply function to traces aggregated by metadata and create new class
        instance with the output.

        If levels arg is provided, func arg must either return an output array
        with an identical number of rows as the input or a single row. Outputs
        must have a consistent shape across all other axes.

        If levels arg is not provided, func accepts a single datum as input
        and returns another datum with a constant shape.

        Args:
            func (function):
                traces_in.shape = (N samples_in_group, N timepoints)
                func(traces_in, *args, *kwargs) = trace_out.
            levels (list, optional):
                Metadata level names that should be collapsed. Data indices
                with identical metadata except along axes specified by levels
                arg will be grouped together prior to applying func arg.
                Defaults to "None", in which case no grouping is performed.
            *args:
                Arguments, passed to func arg.
            **kwargs:
                Keyword arguments, passed to func arg.

        Returns:
            (TimeSeries):
                New instance with traces attribute modified by func arg after
                grouping by cols arg.
        """
        # group traces by key and get corresponding indices as list
        sub_meta = self.meta.sort_index(inplace=False)
        if levels is not None:
            me = list(sub_meta.index.names)
            groups = [i for i in me if i not in levels + [self._TRIAL]]
            sub_meta = sub_meta.groupby(level=groups)[[self._INDEX]].agg(
                lambda x: list(x))

        # apply func to each set of indices in data
        sub_data = np.concatenate([
            np.array(
                func(self.data[i], *args, **kwargs), ndmin=self.axes + 1)
            for i in sub_meta[self._INDEX].tolist()], axis=0)
        return self._safe_copy(sub_data, sub_meta)

    def grouped_control(
            self,
            cont,
            func,
            label: str = "label",
            group: str = "group",
            ignore: list = None
    ):
        """
        Create a new dataframe that pairs each unique grouping of specified
        metadata levels with a corresponding reference set in another control
        instance.

        Returned dataframe has the following structure:

                                 _index   label  _index_other  label_other
            group ... level_n
            0     ... 1          [1, 2]  [a, b]     [1, 2, 3]    [a, q, b]
                      2          [3, 4]  [c, d]     [4, 5, 6]    [r, c, f]
                      3          [5, 6]  [e, f]     [7, 8, 9]    [e, v, d]
            1     ... 1          [7, 8]  [b, a]     [1, 2, 3]    [a, q, b]
                      2         [9, 10]  [r, v]     [4, 5, 6]    [r, c, f]
                      3        [11, 12]  [s, q]     [7, 8, 9]    [e, v, d]

        Wherein each row describes a comparison within metadata values
        specified by the hierarchical index. The first level of this index
        identifies the group in instance to be compared against the entirety of
        the control. The columns describe the following:
        -   _index: Indices along first axis of instance data attribute.
        -   label : Corresponding labels from instance.
        -   _index_other: As above, for entire control.
        -   label_other : As above, for entire control.

        Args:
            cont (CoreDataset):
                Instance to use as reference.
            func ():
                Pass.
            label (str, optional):
                Multiindex level name to use as label for each index in data
                attribute. Must be present in both instance and cont arg
                hierarchical indices.
                Defaults to "label", in which case each data index gets a
                unique label.
            group (str, optional):
                Multiindex level name by which to group data indices into
                subsets for comparison against "cont" arg.
                Defaults to "group", in which case all data indices in self are
                considered part of the same group
            ignore (list, optional):
                Metadata level names to ignore when identifying valid
                comparisons.
                Defaults to "None", in which case comparisons are made within
                all metadata levels outside of those specified by "label" and
                "group" args.

        Returns:
            meta_tot (pd.DataFrame):
                Hierarchically indexed dataframe pairing indices and labels in
                self metadata to a corresponding reference set in control
                instance.
        """
        cont = self if cont is None else cont

        meta_self = self.meta.copy()
        meta_cont = cont.meta.copy()

        # remove metadata levels to ignore
        if ignore is not None:
            meta_cont.index = meta_cont.index.droplevel(ignore)

        # add missing index levels
        if label not in list(meta_self.index.names):
            meta_self = meta_self.assign(
                **{label: np.arange(len(self))}).set_index(label, append=True)
        if label not in list(meta_cont.index.names):
            meta_cont = meta_cont.assign(
                **{label: np.arange(len(cont))}).set_index(label, append=True)
        if group not in list(meta_self.index.names):
            meta_self = meta_self.assign(
                **{group: 0}).set_index(group, append=True)
        if group not in list(meta_cont.index.names):
            meta_cont = meta_cont.assign(
                **{label: 0}).set_index(label, append=True)

        # find shared metadata levels
        meta_self = meta_self.reset_index(label, drop=False)
        meta_cont = meta_cont.reset_index(label, drop=False)
        levels = list(set(meta_self.index.names) & set(meta_cont.index.names))

        # remove trial and label levels before dataframe.groupby operation
        levels = [i for i in levels if i not in (self._TRIAL, label)]

        # duplicate control set comparison for each unique group
        meta_cont.index = meta_cont.index.droplevel(group)
        groups = meta_self.index.unique(level=group).to_list()
        meta_cont = meta_cont.assign(**{group: [groups] * meta_cont.shape[0]})
        meta_cont = meta_cont.explode(group).set_index(group, append=True)

        # create lists of labels and indices to compare for each group
        meta_self = meta_self.groupby(level=levels)[
            [label, self._INDEX]].agg(lambda x: list(x))
        meta_cont = meta_cont.groupby(level=levels)[
            [label, self._INDEX]].agg(lambda x: list(x)).rename(
            columns={
                label: label + self._OTHER,
                self._INDEX: self._INDEX + self._OTHER})

        # join metadata dataframes into a single dataframe
        meta_tot = pd.concat(
            [meta_self, meta_cont], axis=1, ignore_index=False, join="inner")
        meta_tot = meta_tot.apply(
            lambda x: func(
                self.data[x[self._INDEX]],
                np.array(x[label]),
                cont.data[x[self._INDEX + self._OTHER]],
                np.array(x[label + self._OTHER])), axis=1)
        meta_tot = meta_tot.reset_index(drop=False).melt(
            id_vars=levels, var_name=label, value_name="value")
        meta_tot = meta_tot.sort_values(
            [group, label]).explode("value").reset_index(drop=True)
        return meta_tot.replace(
            ["NaN", "nan", "None", None, np.nan, pd.NaT], pd.NA).dropna()
