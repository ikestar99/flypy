#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:00:00 2024
@author: ike
"""


import pickle
import numpy as np
import pandas as pd
import operator as oper
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score

from flypy.datasets.timeseries import TimeSeries
from flypy.datasets.featurevector import FeatureVector


OPERATIONS = {
    "<": oper.lt,
    ">": oper.gt,
    "<=": oper.le,
    ">=": oper.ge,
    "==": oper.eq,
    "!=": oper.ne
}


def load_raw_dataframe(
        file: str,
        col_dict: str = None,
        cols_include: list = None,
        cols_explode: list = None,
        cols_sorting: list = None,
):
    """
    Load and process trial data from a pickle file.

    NOTE: threshold arg currently only supports binary comparisons to

    Args:
        file (str):
            Path to the pickle file.
        col_dict (str, optional):
            Key to raw data DataFrame if pickle file contains a dictionary.
        cols_include (list, optional):
            Column names to include in data.
            Defaults to "None", in which case all columns are kept.
        cols_explode (list, optional):
            Column names to explode along first axis of column values. Within
            each row, all identified columns should contain an unpackable
            iterable with the same length along the first or only axis.
            Explosion unpacks this first dimension across columns such that
            each row is now a single index along the first dimension with
            non-exploded values duplicated.
            Defaults to "None", in which case no explosion is performed.
        cols_sorting (list, optional):
            Column names by which to sort the data. Columns are sorted in arg
            index order.
            Defaults to "None", in which case sort is not performed.

    Returns:
        pd.DataFrame:
            The loaded and pre-processed data.
    """
    with open(file, "rb") as f:
        data = pickle.load(f)

    data = data if col_dict is None else data[col_dict]
    data = data if cols_include is None else data[cols_include]
    data = data if cols_explode is None else data.explode(cols_explode)
    data = data if cols_sorting is None else data.sort_values(
        by=cols_sorting, axis=0)
    return data.reset_index(drop=True)


def load_analyzed_data(
        file: str,
):
    with open(file, "rb") as f:
        data = pickle.load(f)

    return data


def save_analyzed_data(
        file: str,
        data
):
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def filter_format_dataframe(
        data: pd.DataFrame,
        date_format: list = None,
        label_format: list = None,
        binary_thresh: list = None,
        include_filter: list = None,
        cols_sorting: list = None,
        cols_subset: dict = None
):
    """
    Filter and threshold DataFrame by column values.

    NOTE: threshold arg currently only supports binary comparisons to single
    scalar. Filter data directly for more complex behavior

    Args:
        data (pd.DataFrame):
            Data to filter and threshold.
        date_format (list, optional):
            Parameters used to convert a column of str dates to datetime
            objects. List has the following 4 values:
            -   0 (str):
                    Column name containing dates as strings
            -   1 (slice):
                    Indices of string dates from which to use as date template.
            -   2 (str):
                    Format code used to convert string to datetime object. See
                    link: https://docs.python.org/3/library/datetime.html#
                    format-codes
            -   3 (str):
                    Column name to fill with elapsed day.
            This analysis will add a column representing elapsed days with
            respect to the earliest date.
            Defaults to "None", in which case date column is not modified.
        label_format (list, optional):
            Parameters used to convert a column in data to a numeric ground
            truth label. Unique values in converted column are sorted,
            filtered, and mapped to a zero-indexed integer ground truth label
            in ascending order. List has the following 2 values:
            -   0 (str):
                    Column name containing labels to convert to numeric.
            -   1 (str):
                    Column name of ground truth labels. If the same as col arg,
                    replace old labels with new numeric labels.
            Defaults to "None", in which case label column is not modified.
        binary_thresh (list, optional):
            Parameters used to threshold data by column values. Each element in
            list is a tuple with the following three values:
            -   0 (str):
                    Column name on which to perform filtering.
            -   1 (str):
                    Binary operator with which to threshold column. Valid
                    entries are "<", ">", "<=", ">=", "==", "!=".
            -   2 (scalar):
                    Scalar against which to perform thresholding.
            Data is filtered to rows for which row[col] func ref is True.
            Defaults to "None", in which case no thresholding is performed.
        include_filter (list):
            Parameters used to filter data by column values. Filter is
            performed such that data only includes rows where column value is
            in a pre-determined set. Each element in list is a tuple that
            defines a filter with the following two values:
            -   0 (str):
                    Column name to filter.
            -   1 (iterable):
                    Column values to keep.
            Defaults to "None", in which case no filtering is performed.
        cols_sorting (list, optional):
            Column names by which to sort the data. Columns are sorted in arg
            index order.
            Defaults to "None", in which case sort is not performed.
        cols_subset (dict, optional):
            Parameters used to group data into intervals based on a numerical
            column. Contains the following pairings:
            -   key (str):
                    Column name of interval to add.
            -   item (list):
                -   0 (str):
                        existing column name on which to apply interval.
                -   1 (int):
                        interval.
            To add column of elapsed week, cols_subset={"week": [col_day, 7]}.
            Defaults to "None", in which case no interval columns are added.

    Returns:
        pd.DataFrame:
            The pre-processed data.
    """
    if date_format is not None:
        col, idx, template, col_day = date_format
        dates = data[col].copy().astype(str).str[idx].sort_values()
        dates = pd.to_datetime(dates, format=template)
        days = dates.diff().dt.days.fillna(0).cumsum().astype(int)
        data[col] = dates.sort_index()
        data[col_day] = days.sort_index()
    if label_format is not None:
        col, col_out = label_format
        old_labels = np.unique(data[col].copy().values.flatten())
        encoder = {v: k for k, v in enumerate(old_labels)}
        data[col_out] = np.vectorize(lambda x: encoder[x])(data[col])
    if cols_subset is not None:
        for key, value in cols_subset.items():
            data[key] = data[value[0]].floordiv(value[1])

    binary_thresh = [] if binary_thresh is None else binary_thresh
    for col, func, value in binary_thresh:
        data = data.loc[OPERATIONS[func](data[col].copy(), value)]

    include_filter = [] if include_filter is None else include_filter
    for col, values in include_filter:
        data = data.loc[data[col].isin(values)]

    data = data if cols_sorting is None else data.sort_values(
        by=cols_sorting, axis=0)
    return data.reset_index(drop=True)


def dataframe_to_time_dataset(
        data: pd.DataFrame,
        col_trace: str,
        cols_meta: list,
        timepoints,
        axis_split: dict = None,
        expand: list = None,
        f_norm=None,
        agg_trials: list = None,
        clip=None
):
    """
    Converts data from a DataFrame to a TimeSeries dataset.

    Args:
        data (pd.DataFrame):
            Data to convert to TimeSeries.
        col_trace (str):
            Column name of trace data.
        cols_meta (list):
            Column names to include as metadata variables.
        timepoints (iterable):
            Time axis ticks of trace data. Must be 1D with same length as time
            axis of trace data. Equivalent to
            np.arange(start_time, stop_time, 1 / Hz).
            Arg should reflect time axis after any normalization is performed.
        axis_split (dict, optional):
            Parameters used to split an axis of trace data into two axes.
            Contains the following pairings:
            -   key (int):
                    Axis in trace data array to split.
            -   item (int):
                    Length of extracted axis. Axis will be split in-place
            For each dict pairing, trace_data.shape is modified as such:
            [..., n_features x repeat, ...] --> [..., n_features, repeat, ...]
            where n_features x repeat is the length of trace_data.shape[key].
            Useful if separate features are concatenated in a single axis.
            Defaults to "None", in which case no split is performed.
        expand (list, optional):
            Passed to TimeSeries.__init__. If present and axis_split arg is not
            None, expand should use axis labels expected after all axis splits
            are performed.
            Defaults to "None", in which case no expansion is performed.
        f_norm (func, optional):
            Custom function used to normalize each trace individually.
            f_func(trace) = trace.
            Defaults to "None", in which case trace data is not normalized.
        agg_trials (list, optional):
                Parameters with which to extract central tendency across
                repeated trials. Contains the following 3 values:
                -   0 (list):
                        Metadata multiindex column names by which to group
                        traces into sets that will be aggregated.
                -   1 (np.ufunc):
                        Function with which to perform aggregation.
                -   3 (dict, optional):
                        Kwargs to pass to aggregation function. Optional.
                Passed to TimeSeries.apply_2d_function.
                Defaults to "None", in which case repeated trials are not
                aggregated.
        clip (np.ndarray, optional):
            Indices of timepoints to which TimeSeries traces are clipped.
            Defaults to "None", in which case traces are not clipped.

    Returns:
        data (TimeSeries):
            The TimeSeries dataset.
    """
    traces = np.array(data[col_trace].to_list())
    axis_split = {} if axis_split is None else axis_split
    axis_split = {
        k if k > 0 else traces.ndim + k: v for k, v in axis_split.items()}
    for i, axis in enumerate(sorted(axis_split)):
        length = axis_split[axis]
        axis = axis + i
        old_shape = list(traces.shape)
        n = old_shape[axis] // length
        new_shape = old_shape[:axis] + [n, length] + old_shape[axis + 1:]
        traces = np.take(traces, indices=np.arange(n * length), axis=axis)
        traces = np.reshape(traces, newshape=new_shape)

    meta = {key: data[key].to_list() for key in cols_meta}
    data = TimeSeries(
        traces, timepoints, trace_names=meta, expand=expand,
        f_norm=f_norm, agg_trials=agg_trials)
    data = data if clip is None else data.clip_traces(clip)
    return data


def timeseries_to_vector_dataset(
        data: TimeSeries,
        f_feature,
        cols_source: list,
        *args,
        **kwargs
):
    """
    Convert a TimeSeries dataset into a FeatureVector dataset.

    Args:
        data (TimeSeries):
            Data to convert.
        f_feature (func):
            Scalar-valued function with which to extract feature. Takes a 1D
            np.ndarray as input. Passed to TimeSeries.apply_scalar_function.
        cols_source (list):
            Metadata multiindex column name(s) by which to group features
            into vectors. Act as corresponding labels for feature vectors.
            Passed to TimeSeries.get_vector_form.
        *args (positional arguments):
            Passed to TimeSeries.get_vector_form.
        **kwargs (keyword arguments)
            Passed to TimeSeries.get_vector_form.

    Returns:
        (FeatureVector):
            Data converted to feature vectors.
    """
    _col_temp = "_foo"
    data = data.apply_scalar_function(_col_temp, f_feature, *args, **kwargs)
    vectors, labels = data.get_vector_form(cols_source, _col_temp)
    return FeatureVector(vectors, labels)


def correlation_over_time(
        data: TimeSeries,
        refs: TimeSeries,
        col_label: str,
        col_group: str,
        cols_source: list = None,
        label_fill: int = -1,
        col_value: str = "correlation",
        col_sep: str = "separability"
):
    """
    Compute pairwise correlations across time between and within labels.

    Args:
        data (TimeSeries):
            Data to correlate across groups.
        refs (TimeSeries):
            Filtered subset of data arg to use as reference against which all
            groups in data arg will be correlated.
        col_label (str):
            Metadata multiindex column name to use as label of each trace when
            computing within and between label correlations.
            Passed to TimeSeries.pairwise_corr_against_reference.
        col_group (str):
            Metadata multiindex column name to use to group traces into subsets
            before correlating each individual subset against refs arg.
            Passed to TimeSeries.pairwise_corr_against_reference.
        cols_source (list, optional):
            Metadata multiindex column name(s) to use as trace source during
            correlation. Traces are correlated within sources only.
            Passed to TimeSeries.pairwise_corr_against_reference.
            Defaults to "None", in which case all traces are treated as if they
            come from the same source.
        label_fill (int, optional):
            Integer used as label of comparisons across traces with different
            labels. Must not be present in the set of labels in data or refs
            args.
            Defaults to "-1".
        col_value (str, optional):
            Name of column in returned DataFrames that contains correlation
            values.
            Defaults to "correlation".
        col_sep (str, optional):
            Name of column in returned DataFrames that contains correlation
            separability.
            Defaults to "separability".

    Returns:
        (tuple):
            contains the following 3 values:
            -   within_df (pd.DataFrame):
                    Correlations within labels across sources for each group in
                    data arg against refs arg. Has the following columns:
                    -   col_group
                    -   cols_source if not None
                    -   col_label
                    -   col_value
                    where each row is a correlation between two traces of the
                    same label and set of sources, one from the specified group
                    in data arg and another from refs arg.
            -   across_df (pd.DataFrame):
                    Same structure as within_df return, different number of
                    rows. col_label has one of two values in each row:
                    -   0: correlation value, comparison across labels.
                    -   1: correlation value, comparison within labels.
            -   sep_df (pd.DataFrame)
                    Difference in average correlation within vs between labels
                    for each group and sources set. Has the following columns:
                    -   col_group
                    -   cols_source if not None
                    -   col_sep
                    where each row is a difference between the mean correlation
                    within and between labels for the specified metadata set.
    """
    cols_source = [] if cols_source is None else cols_source
    cols_apply = [col_group] + cols_source + [col_label]
    within_df = data.pairwise_corr_against_reference(
        other=refs, col_label=col_label, col_group=col_group,
        cols_source=cols_source).reset_index().melt(
        id_vars=cols_apply[:-1], var_name=col_label, value_name=col_value)
    within_df = within_df.sort_values(cols_apply).explode(
        col_value).reset_index(drop=True)
    within_df = within_df.replace('NaN', pd.NA).dropna(axis=0)
    across_df = within_df.copy()
    across_df[col_label] = (across_df[col_label] != label_fill).astype(int)
    sep_df = across_df.copy().groupby(cols_apply).mean().reset_index().pivot(
        index=cols_apply[:-1], columns=col_label, values=col_value)
    sep_df = pd.DataFrame(sep_df[1] - sep_df[0], columns=[col_sep])
    sep_df = sep_df.sort_index().reset_index(drop=False)
    within_df = within_df.loc[within_df[col_label] != label_fill]
    return within_df, across_df, sep_df.replace('NaN', pd.NA).dropna(axis=0)


def cosine_similarity_over_time(
        data: FeatureVector,
        refs: FeatureVector,
        col_label: str,
        col_group: str,
        mapping: dict,
        label_fill: int = -1,
        col_value: str = "cosine",
        col_sep: str = "separability"
):
    """
    Compute pairwise cosine similarities across time between and within labels.

    Args:
        data (FeatureVector):
            Data to correlate across groups.
        refs (FeatureVector):
            Filtered subset of data arg to use as reference against which all
            groups in data arg will be compared.
        col_label (str):
            Metadata multiindex column name to use as label of each feature
            vector when computing within and between label similarities.
        col_group (str):
            Metadata multiindex column name to use to group feature vectors
            into subsets before computing cosine similarities against refs arg.
            Passed to FeatureVector.pairwise_cosine_against_reference.
        mapping (dict)
            Maps index in second axis of FeatureVector labels attribute to
            corresponding label type. Contains the following two pairings:
            -   key (str):
                    One of two values:
                    -   col_label
                    -   col_group
            -   item (int)
                    Index of key along second axis of FeatureVector
                    labels attribute.
        label_fill (int, optional):
            Integer used as label of comparisons across feature vectors with
            different labels. Must not be present in the set of labels in data
            or refs args.
            Defaults to "-1".
        col_value (str, optional):
            Name of column in returned DataFrames that contains cosine
            similarity values.
            Defaults to "cosine".
        col_sep (str, optional):
            Name of column in returned DataFrames that contains cosine
            separability.
            Defaults to "separability".

    Returns:
        (tuple):
            contains the following 3 values:
            -   within_df (pd.DataFrame):
                    Similarities within labels across sources for each group in
                    data arg against refs arg. Has the following columns:
                    -   col_group
                    -   col_label
                    -   col_value
                    where each row is a similarity between two vectors of the
                    same label and set of sources, one from the specified group
                    in data arg and another from refs arg.
            -   across_df (pd.DataFrame):
                    Same structure as within_df return, different number of
                    rows. col_label has one of two values in each row:
                    -   0: similarity value, comparison across labels.
                    -   1: similarity value, comparison within labels.
            -   sep_df (pd.DataFrame)
                    Difference in average correlation within vs between labels
                    for each group and sources set. Has the following columns:
                    -   col_group
                    -   col_sep
                    where each row is a difference between the mean similarity
                    within and between labels for the specified metadata set.
    """
    within_df = data.pairwise_cos_against_reference(
        other=refs, idx_l=mapping[col_label], idx_g=mapping[col_group],
        col_group=col_group)
    within_df = within_df.reset_index().melt(
        id_vars=col_group, var_name=col_label, value_name=col_value)
    within_df = within_df.sort_values(
        [col_group, col_label]).explode(col_value).reset_index(drop=True)
    within_df = within_df.replace('NaN', pd.NA).dropna(axis=0)
    across_df = within_df.copy()
    across_df[col_label] = (across_df[col_label] != label_fill).astype(int)
    sep_df = across_df.copy().groupby([col_group, col_label]).mean()
    sep_df = sep_df.reset_index().pivot(
        index=col_group, columns=col_label, values=col_value)
    sep_df = pd.DataFrame(sep_df[1] - sep_df[0], columns=[col_sep])
    sep_df = sep_df.sort_index().reset_index(drop=False)
    within_df = within_df.loc[within_df[col_label] != label_fill]
    return within_df, across_df, sep_df.replace('NaN', pd.NA).dropna(axis=0)


def encode_with_dim_reducer(
        data: FeatureVector,
        refs: FeatureVector,
        reducer,
        mapping: dict,
        dim_name: str = "components",
        var_value: str = "variance ratio"
):
    """
    Compute pairwise cosine similarities across time between and within labels.

    Args:
        data (FeatureVector):
            Data to correlate across groups.
        refs (FeatureVector):
            Filtered subset of data arg to use as reference against which all
            groups in data arg will be compared.
        reducer:
            Instantiated dimensionality reducing object with which to encode
            extracted feature vectors. An example is PCA.
        mapping (dict)
            Maps index in second axis of FeatureVector labels attribute to
            corresponding label type. Contains the following pairings:
            -   key (str):
                    Name of label encoded in index.
            -   item (int)
                    Index of key along second axis of FeatureVector
                    labels attribute.
        dim_name (str, optional):
            Each encoded feature vector axis is named "dim_name i" where i is
            the index of the encoded feature in each feature vector.
            Defaults to "component".
        var_value (str, optional):
            Column name of variance ratio per encoded dim.
            Defaults to "variance ratio".

    Returns:
        (tuple):
            contains the following 3 values:
            -   reduced_df (pd.DataFrame):
                    Encoded feature vectors. Has the following columns:
                    -   col_group
                    -   col_label
                    -   "dim name 'i'" for each encoded component
                    where each row is a similarity between two vectors of the
                    same label and set of sources, one from the specified group
                    in data arg and another from refs arg.
            -   variance_df (pd.DataFrame):
                    Fraction of variance captured per encoded dimension.
                    Contains the following columns:
                    -   dim_name
                    -   var_value
    """
    inverse = {v: k for k, v in mapping.items()}
    data = np.concatenate(
        (data.labels, data.encode(refs.fit_model(reducer)).vectors), axis=1)
    n_components = data.shape[1] - len(mapping)
    cols = [inverse[i] for i in range(len(mapping))]
    cols += [f"{dim_name} {i}" for i in range(n_components)]
    reduced_df = pd.DataFrame(data, columns=cols)
    variance_df = np.stack(
        (np.arange(n_components), reducer.explained_variance_ratio_), axis=1)
    variance_df = pd.DataFrame(variance_df, columns=[dim_name, var_value])
    return reduced_df, variance_df


def classification_grid_search(
        data: TimeSeries,
        func_dict: dict,
        model_dict: dict,
        cols_source: list,
        col_label: str,
        k_fold: int = 5,
        normalize: bool = False
):
    """
    Find feature, model combination that optimize classification accuracy.
    Args:
        data (TimeSeries):
            Data from which to extract features vectors.
        func_dict (dict):
            Functions with which to extract features. Contains the following
            pairings:
            -   key (str):
                    Name of function.
            -   item (np.ufunc):
                    Function used to extract feature from each trace.
        model_dict (dict):
            Classification models with which to classify feature vectors.
            Contains the following pairings:
            -   key (str):
                    Name of model.
            -   item (list):
                    -   0 (model):
                            Model class
                    -   1 (dict, optional):
                            kwargs.
                            Defaults to {}.
                    model = item[0](**item[1])
        cols_source (list):
            Column names by which to group features into vectors. Each vector
            has a unique combination of values for each level described in arg.
        col_label (str):
            Column name used as ground truth for each vector.
        k_fold (int):
            Number of folds with which to train each model.
            Defaults to "5".
        normalize (bool):
            If True, normalize each feature across all vectors to 0 mean and
            unit variance.
            Defaults to "False"

    Returns:
        (tuple):
            Contains the following three values:
            -   r_accuracy (pd.DataFrame):
                    Index is model name, columns are feature function names.
                    r_accuracy.loc[m, f] = k-fold average accuracy of model m
                    when trained on features f extracted from grouped traces.
            -   model (str):
                    Name of model with max accuracy.
            -   feature (str)
                    Name of feature extraction function with max accuracy.

    """
    def _k_fold_accuracy(_model, _vectors):
        _model = _model[0]() if len(_model) == 1 else _model[0](**_model[1])
        output = np.array([
            accuracy_score(*v.classify(t.fit_model(_model)))
            for t, v in _vectors])
        return np.mean(output)

    r_accuracy = []
    for key, func in func_dict.items():
        vectors = timeseries_to_vector_dataset(
            data, f_feature=func, cols_source=cols_source).scale()
        vectors.labels = vectors.labels[:, [cols_source.index(col_label)]]
        vectors = vectors.scale() if normalize else vectors
        vectors = [
            [unit_t, unit_v] for unit_t, unit_v
            in vectors.k_fold_split(k_fold)]
        acc = {k: _k_fold_accuracy(m, vectors) for k, m in model_dict.items()}
        acc = pd.DataFrame.from_dict(acc, orient="index", columns=[key])
        r_accuracy += [acc]

    r_accuracy = pd.concat(r_accuracy, axis=1, ignore_index=False)
    model, feature = r_accuracy.stack().index[np.argmax(r_accuracy.values)]
    return r_accuracy, model, feature


def source_label_variance(
        data: TimeSeries,
        f_feature,
        col_label: str,
        col_group: str = None,
        cols_source: list = None,
        f_labels: np.ufunc = np.mean,
):
    """
    Identify sources in a given column with the greatest CV across labels. CV,
    the coefficient of variation, is defined as the ratio σ/μ.

    The following pseudocode describes the process of computing central
    tendency and CV for each source and group.

    # fake data from group i source j
    data = np.ones((N_labels, N_timepoints))
    data = f_feature(data, axis=1)  # data.shape = (N_labels,)
    data_out[0].iloc[i, j] = f_labels(data)  # scalar central tendency
    data_out[1].iloc[i, j] = np.std(data) / np.mean(data)  # scalar CV

    NOTE: To extract a signal-to-noise ratio feature, or a feature on a slice
    write a custom function
    that defines this parameter directly, like so:
    source_label_variance(
        ...,
        f_extraction=lambda x: np.mean(x[idx_signal]) / np.mean(x[idx_noise]),
        ...,
        idx_signal=None
    )

    Args:
        data (TimeSeries):
            Data from which to extract salience
        f_feature (func):
            Scalar-valued function with which to extract feature. Takes a 1D
            np.ndarray as input. Passed to TimeSeries.apply_scalar_function.
        col_label (str):
            Column name to use as label.
        col_group (str, optional):
            Column name to use as group identity for each trial. A set of CVs
            will be computed independently for each group.
            Defaults to "None", in which case all traces are treated as part
            of the same group.
        cols_source (list, optional):
            Column name(s) to use as source(s) from which tuning is computed.
            cols_source = column_of_electrode_idx to compute electrode tunings.
            Defaults to "None", in which case all features are treated as if
            they came from the same source.
        f_labels (np.ufunc, optional):
            Scalar-valued function with which to extract central tendency
            across labels. Passed to pd.DataFrame.groupby.agg.
            Defaults to "np.mean".

    Returns:
        (tuple):
            Contains the following two values:
            -   (pd.DataFrame):
                    Central tendency of features across labels. Has the
                    following columns:
                    -   col_group if not "None"
                    -   cols_source if not "None"
                    -   "value"
            -   1 (pd.DataFrame):
                    Central tendency of features across labels. Same shape and
                    structure as above.
    """
    _col_holder_func = "value"
    _col_holder_group = "_group"
    if col_group is None:
        data.meta[_col_holder_group] = 0
        data = data.add_index([_col_holder_group], append=True)
        col_group = _col_holder_group

    cols_source = [] if cols_source is None else cols_source
    cols_apply = [col_group] + cols_source + [col_label]
    data = data.apply_scalar_function(
        _col_holder_func, lambda x: f_feature(x)).meta.copy()
    data = data.groupby(level=cols_apply[:-1])
    return (
        data.agg(f_labels).reset_index(),
        data.std().div(data.mean()).reset_index())


# def train_k_classifiers(
#         train_vectors,
#         classifier,
#         k,
#         idx_l: int = 0,
#         idx_g: int = None,
#         **kwargs
# ):
#     single = True if k < 2 else False
#     k = max(2, k)
#     splits = [
#         (unit_t, unit_v) for unit_t, unit_v in train_vectors.k_fold_split(k)]
#     splits = splits[:1] if single else splits
#     models = [
#         units[0].fit_model(classifier(**kwargs), idx_l) for units in splits]
#     fold_df = [
#         [["train"] + units[0].classify(models[i], idx_l, idx_g),
#          ["validation"] + units[1].classify(models[i], idx_l, idx_g),
#          ["chance"] + units[1].shuffle_within().classify(
#              models[i], idx_l, idx_g)]
#         for i, units in enumerate(splits)]
#     fold_df = [v for s in fold_df for v in s]
#     columns = ["mode", "label", "prediction"]
#     columns = columns if idx_g is None else columns + ["group"]
#     group = ["mode"] if idx_g is None else ["mode", "group"]
#     fold_df = pd.DataFrame(data=fold_df, columns=columns).explode(columns[1:])
#     fold_df = fold_df.groupby(group)[["label", "prediction"]].agg(
#         lambda x: list(x)).apply(
#         lambda x: pd.Series(
#             accuracy_score(x["label"], x["prediction"]), index=["accuracy_score"]), axis=1)
#     return models, fold_df[["accuracy_score"]].reset_index()
#
#
# def test_k_classifiers(
#         test_vectors,
#         classifiers,
#         idx_l: int = 0,
#         idx_g: int = None
#
# ):
#     classifiers = classifiers if type(classifiers) is list else [classifiers]
#     test_df = [
#         [["test"] + test_vectors.classify(c, idx_l, idx_g),
#          ["chance"] + test_vectors.shuffle_within().classify(c, idx_l, idx_g)]
#         for c in classifiers]
#     test_df = [v for s in test_df for v in s]
#     columns = ["mode", "label", "prediction"]
#     columns = columns if idx_g is None else columns + ["group"]
#     group = ["mode"] if idx_g is None else ["mode", "group"]
#     test_df = pd.DataFrame(data=test_df, columns=columns).explode(columns[1:])
#     test_df = test_df.groupby(group)[["label", "prediction"]].agg(
#         lambda x: list(x)).apply(
#         lambda x: pd.Series(
#             accuracy_score(x["label"], x["prediction"]), index=["accuracy_score"]), axis=1)
#     return test_df[["accuracy_score"]].reset_index()
#
#
# def train_test_frozen_classifier(
#         all_vectors,
#         classifier,
#         idx_l,
#         idx_g,
#         start=1,
#         **kwargs
# ):
#     train_test_df = [
#         [g, accuracy_score(
#             *unit_v.classify(
#                 unit_t.fit_model(classifier(**kwargs), idx_l), idx_l))]
#         for g, unit_t, unit_v
#         in all_vectors.leave_last_out_split(idx_g, start)]
#     return pd.DataFrame(data=train_test_df, columns=["group", "accuracy_score"])


class ControlEncoder:
    """
    Simple control encoder class.
    """

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    @staticmethod
    def transform(X, *args, **kwargs):
        """
        Transforms the input data.

        Args:
            X: Input data.

        Returns:
            Input data (unchanged).
        """
        return X
