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

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score

from flypy.datasets.timeseries import TimeSeries
from flypy.datasets.featurevector import FeatureVector
from flypy.datasets.wrappers import TimeVectorClassificationWrapper
from flypy.models.rnns import LangClassifierConvLSTM
from flypy.models.wrappers import RNNClassifierWrapper


OPERATIONS = {
    "<": oper.lt,
    ">": oper.gt,
    "<=": oper.le,
    ">=": oper.ge,
    "==": oper.eq,
    "!=": oper.ne
}


def load_analyzed_data(
        file: str,
):
    """
    Load pickled object.

    Args:
        file (str):
            Path to pickle file.

    Returns:
        (dict)
            Unpickled data.
    """
    with open(file, "rb") as f:
        return pickle.load(f)


def save_analyzed_data(
        file: str,
        data
):
    """
    Save object in pickle file.
    Args:
        file (str):
            Path to pickle file.
        data (object):
            Data to save.
    """
    with open(file, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_raw_dataframe(
        file: str,
        col_dict: str = None,
        include: list = None,
        explode: list = None,
        sortby: list = None,
):
    """
    Load and process trial data from a pickle file.

    NOTE: threshold arg currently only supports binary comparisons to

    Args:
        file (str):
            Path to the pickle file.
        col_dict (str, optional):
            Key to raw data DataFrame if pickle file contains a dictionary.
        include (list, optional):
            Column names to include in data.
            Defaults to "None", in which case all columns are kept.
        explode (list, optional):
            Column names to explode along first axis of column values. Within
            each row, all identified columns should contain an unpackable
            iterable with the same length along the first or only axis.
            Explosion unpacks this first dimension across columns such that
            each row is now a single index along the first dimension with
            non-exploded values duplicated.
            Defaults to "None", in which case no explosion is performed.
        sortby (list, optional):
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
    data = data if include is None else data[include]
    data = data if explode is None else data.explode(explode)
    data = data if sortby is None else data.sort_values(by=sortby, axis=0)
    return data.reset_index(drop=True)


def filter_format_dataframe(
        data: pd.DataFrame,
        include_filter: list = None,
        date_format: list = None,
        label_format: list = None,
        binary_thresh: list = None,
        sortby: list = None,
        subset: dict = None
):
    """
    Filter and threshold DataFrame by column values.

    NOTE: threshold arg currently only supports binary comparisons to single
    scalar. Filter data directly for more complex behavior

    Args:
        data (pd.DataFrame):
            Data to filter and threshold.
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
        sortby (list, optional):
            Column names by which to sort the data. Columns are sorted in arg
            index order.
            Defaults to "None", in which case sort is not performed.
        subset (dict, optional):
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
    # include only a subset of values in a given metadata level
    include_filter = [] if include_filter is None else include_filter
    for col, values in include_filter:
        data = data.loc[data[col].isin(values)]

    # create elapsed day metadata level from string date metadata level
    if date_format is not None:
        col, idx, template, col_day = date_format
        dates = data[col].copy().astype(str).str[idx].sort_values()
        dates = pd.to_datetime(dates, format=template)
        days = dates.diff().dt.days.fillna(0).cumsum().astype(int)
        data.loc[:, col] = dates.sort_index()
        data.loc[:, col_day] = days.sort_index()

    # convert label metadata to 0-indexed integer ground truths
    if label_format is not None:
        col, col_out = label_format
        old_labels = np.unique(data[col].copy().values.flatten())
        encoder = {v: k for k, v in enumerate(old_labels)}
        data.loc[:, col_out] = np.vectorize(lambda x: encoder[x])(data[col])

    # create new metadata level by parsing existing metadata into intervals
    if subset is not None:
        for key, value in subset.items():
            data.loc[:, key] = data[value[0]].floordiv(value[1])

    # apply binary threshold by metadata level value
    binary_thresh = [] if binary_thresh is None else binary_thresh
    for col, func, value in binary_thresh:
        data = data.loc[OPERATIONS[func](data[col].copy(), value)]

    # sort values
    data = data if sortby is None else data.sort_values(
        by=sortby, axis=0)
    return data.reset_index(drop=True)


def dataframe_to_time_dataset(
        data: pd.DataFrame,
        col_trace: str,
        timepoints,
        axis_split: dict = None,
        expand: list = None,
        f_norm=None,
):
    """
    Converts data from a DataFrame to a TimeSeries dataset.

    Args:
        data (pd.DataFrame):
            Data to convert to TimeSeries.
        col_trace (str):
            Column name of trace data.
        timepoints (iterable):
            Time axis ticks of trace data after any normalization is performed.
            Equivalent to np.arange(start_time, stop_time, 1 / Hz).
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
            normalized_trace = f_func(trace).
            Defaults to "None", in which case trace data is not normalized.

    Returns:
        data (TimeSeries):
            The TimeSeries dataset.
    """
    traces = np.array(data[col_trace].to_list())

    # unpack axes with tiled metadata levels
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

    # create TimeSeries and normalize traces if applicable
    meta = [i for i in data.columns if i != col_trace]
    data = TimeSeries(traces, data[meta], timepoints, expand=expand)
    data = data if f_norm is None else data.apply_function(f_norm)
    return data


def latent_representation_over_time(
        data: FeatureVector,
        model,
        label: str = None,
        cutoff: float = 0.8
):
    """
    Perform feature vector dimensionality reduction using an encoding model.

    Args:
        data (FeatureVector):
            Data to correlate across groups.
        model:
            Instantiated dimensionality reducing object with which to transform
            instance data.
            Passed to FeatureVector.fit_model.
        label (str, optional):
            For supervised encoding: metadata level name to use as vector label
            when fitting model to instance data.
            Passed to FeatureVector.fit_model.
            Defaults to "None", in which case model is unsupervised.
        cutoff (float, optional):
            Cumulative variance used to determine how many components to keep.
            Defaults to "0.8".

    Returns:
        (tuple):
            data (FeatureVector):
                Encoded feature vectors.
            variance (np.ndarray):
                Cumulative variance captured per encoded dimension.
    """
    model = data.fit_model(model, label)
    data = data.transform_model(model)
    variance = np.cumsum(model.explained_variance_ratio_)
    variance = variance[variance <= cutoff]
    data.data = data.data[..., :variance.size]
    return data, variance


def classification_grid_search(
        data: TimeSeries,
        func_dict: dict,
        model_dict: dict,
        levels: list,
        label: str,
        k_fold: int = 10,
        normalize: bool = True
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
            Classification models with which to predict_model feature vectors.
            Contains the following pairings:
            -   key (str):
                    Name of model.
            -   item (list):
                -   0 (model):
                        Model class
                -   1 (dict, optional):
                        Model instantiation kwargs.
        levels (list):
            Metadata level names to collapse into feature vectors.
            Passed to TimeSeries.to_feature_vector.
        label (str):
            Column name used as ground truth for each vector.
        k_fold (int, optional):
            Number of folds with which to train each model.
            Defaults to "1o".
        normalize (bool, optional):
            If True, normalize each feature across all vectors to 0 mean and
            unit variance.
            Defaults to "True".

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
    r_accuracy = []

    # iterate over feature extraction functions
    for key, func in tqdm(func_dict.items(), desc="grid search functions"):
        vectors = data.to_feature_vector(func, levels=levels)
        vectors = vectors.scale() if normalize else vectors
        acc = {k: 0 for k in model_dict}
        for unit_t, unit_v in vectors.k_fold_split(k_fold, label):
            for k, m in model_dict.items():
                m = m[0](**(m + [{}])[1])
                acc[k] += accuracy_score(
                    unit_v[label], unit_v.predict_model(
                        unit_t.fit_model(m, label))) / k_fold

        acc = pd.DataFrame.from_dict(acc, orient="index", columns=[key])
        r_accuracy += [acc]

    r_accuracy = pd.concat(r_accuracy, axis=1, ignore_index=False)
    model, feature = r_accuracy.stack().index[np.argmax(r_accuracy.values)]
    return r_accuracy, model, feature


def feature_salience_over_time(
        data: TimeSeries,
        func,
        source: str,
        levels: list,
        center: np.ufunc = np.mean,
        spread: np.ufunc = np.std
):
    _temp = "_temporary_column"
    levels = [i for i in data.meta.index.names if i not in levels]
    meta_sub = data.extract_statistic(func, _temp)[[_temp]].groupby(
        level=levels).apply(lambda x: pd.Series(
            [center(x), spread(x)], index=["center", "spread"]))
    return meta_sub


def shallow_learning_curves(
        data: FeatureVector,
        model,
        label: str,
        group: str,
        reverse: bool = False,
        k_fold: int = 10,
        step: int = 1,
        test_fraction: float = 0.1
):
    """
    Generate forward and reverse learning curves with linear encoding model.

    Say a dataset includes the following groups: [0, 6, 5, 9, 8, 7, 3, 2, 1, 4]
    shallow_learning_curve(
        ...,
        reverse = True,
        start = 2,
        train_fraction = 0.75,
        test_fraction = 0.2
    )
    will generate a learning curve such that:
        Constant test set -- groups (1, 0) = last 20% of groups, descending
        Training subset 0 -- groups (9, 8) = first 2 remaining groups
            Train on 80%, validate on 20%, test on constant test set
        Training subset 1 -- groups (9, 8, 7)
            Repeat process described above
        ...
        Training subset n -- groups (9, ..., 2) = entire dataset excluding test
            Repeat process described above.

    Args:
        data (FeatureVector):
            Data on which to train, validate, and test models.
        model:
            Instantiated classification linear model object with which to
            predict labels of instance data.
            Passed to FeatureVector.fit_model.
        label (str):
            Metadata level name used as vector ground truth label.
        group (str):
            Metadata level name used to slice training data into buckets.
        reverse (bool, optional):
            If "True", group labels are sorted in descending order before
            segmenting into training buckets and a constant test set.
            Defaults to "False", in which case sort in ascending order.
        k_fold (int, optional):
            Number of folds with which to train each model.
            Defaults to "1o".
        step (int, optional):
            Number of groups to add when identifying the next training subset.
            Defaults to "1", in which case each subsequent subset has one more
            group than the previous
        test_fraction (float, optional):
            Last fraction of group-sorted dataset to use as constant test set.
            Defaults to "0.1", in which case last 10% of sorted groups serve as
            test set for every training subset.

    Returns:
        (np.ndarray):
            shape = (N training subsets, 3, N labels, N predicted labels).
            Second axis corresponds to (train, validation, constant test set).
            Last two axes constitute confusion matrices.
        (np.ndarray, optional):
            shape = (N training subsets, N labels, N features).
            Last two axes constitute weight vectors for trained model if
            defined in model.coef_ attribute, as per sklearn convention. See:
            sklearn.linear_model.LogisticRegression for example.
            Returns "None" if attribute not defined.
    """
    # return confusion matrices, feature salience too if defined by model
    matrices = []
    salience = [] if hasattr(model, "coef_") else None

    # order unique group labels
    groups = np.unique(data[group])
    groups = groups[::-1] if reverse else groups
    split = int((1 - test_fraction) * groups.size)

    # create constant test set = last fraction of group labels
    cont_t = data[{group: groups[split:]}]

    # generate progressively larger training subsets
    start = step * ((k_fold // step) + int((k_fold % step) > 0))
    for i in tqdm(range(start, split, step), desc="shallow curve sets"):
        folds = []
        r_sal = 0
        for unit_t, unit_v in data[{group: groups[:i]}].k_fold_split(
                k_fold, label):

            # train model on train split of current subset fold
            model = unit_t.fit_model(model, label)

            # folds.shape = (3 (train, valid, test), N classes, N predicted)
            folds += [np.stack([
                confusion_matrix(unit_t[label], unit_t.predict_model(model)),
                confusion_matrix(unit_v[label], unit_v.predict_model(model)),
                confusion_matrix(cont_t[label], cont_t.predict_model(model))],
                axis=0)]

            # aggregate salience if relevant to current model
            r_sal += 0 if salience is None else model.coef_

        # save data from current iteration
        matrices += [np.stack(folds, axis=0)]
        salience = salience if salience is None else salience + [r_sal]

    matrices = np.stack(matrices, axis=0)
    salience = salience if salience is None else np.stack(salience, axis=0)
    return matrices, salience


def deep_learning_curves(
        data: FeatureVector,
        label: str,
        group: str,
        times: str,
        save_dir: str,
        reverse: bool = False,
        k_fold: int = 10,
        step: int = 1,
        test_fraction: float = 0.1,
        train_fraction: float = 0.8,
        epochs: int = 50
):
    """
    Generate forward and reverse learning curves with linear encoding model.

    Say a dataset includes the following groups: [0, 6, 5, 9, 8, 7, 3, 2, 1, 4]
    shallow_learning_curve(
        ...,
        reverse = True,
        start = 2,
        train_fraction = 0.75,
        test_fraction = 0.2
    )
    will generate a learning curve such that:
        Constant test set -- groups (1, 0) = last 20% of groups, descending
        Training subset 0 -- groups (9, 8) = first 2 remaining groups
            Train on 80%, validate on 20%, test on constant test set
        Training subset 1 -- groups (9, 8, 7)
            Repeat process described above
        ...
        Training subset n -- groups (9, ..., 2) = entire dataset excluding test
            Repeat process described above.

    Args:
        data (FeatureVector):
            Data on which to train, validate, and test models.
        model:
            Instantiated classification linear model object with which to
            predict labels of instance data.
            Passed to FeatureVector.fit_model.
        label (str):
            Metadata level name used as vector ground truth label.
        group (str):
            Metadata level name used to slice training data into buckets.
        times (str):
            Metadata level name used as vector timepoint.
        reverse (bool, optional):
            If "True", group labels are sorted in descending order before
            segmenting into training buckets and a constant test set.
            Defaults to "False", in which case sort in ascending order.
        k_fold (int, optional):
            TODO: implement k-fold validation
            Number of folds with which to train each model.
            Defaults to "1o".
        step (int, optional):
            Number of groups to add when identifying the next training subset.
            Defaults to "1", in which case each subsequent subset has one more
            group than the previous
        test_fraction (float, optional):
            Last fraction of group-sorted dataset to use as constant test set.
            Defaults to "0.1", in which case last 10% of sorted groups serve as
            test set for every training subset.
        train_fraction (float, optional):
            Fraction of each subset dataset to use for training.
            Defaults to "0.8", in which case last 80% of current subset is used
            for training and the remaining 20% for validation.

    Returns:
        (np.ndarray):
            shape = (N training subsets, 3, N labels, N predicted labels).
            Second axis corresponds to (train, validation, constant test set).
            Last two axes constitute confusion matrices.
        TODO: return latent representation of deep learning model
        (np.ndarray, optional):
            shape = (N training subsets, N labels, N features).
            Last two axes constitute weight vectors for trained model if
            defined in model.coef_ attribute, as per sklearn convention. See:
            sklearn.linear_model.LogisticRegression for example.
            Returns "None" if attribute not defined.
    """
    # order unique group labels
    groups = np.unique(data[group])
    groups = groups[::-1] if reverse else groups
    split = int((1 - test_fraction) * groups.size)
    n_labels = np.unique(data[label]).size

    # create constant test set = last fraction of group labels
    cont_t = TimeVectorClassificationWrapper(
        data=data[{group: groups[split:]}],
        label=label,
        times=times)

    # generate progressively larger training subsets
    r_statistics = []
    start = step * ((k_fold // step) + int((k_fold % step) > 0))
    for i in tqdm(range(start, split, step), desc="deep curve sets"):
        unit_s = TimeVectorClassificationWrapper(
            data=data[{group: groups[:i]}],
            label=label,
            times=times)
        model = LangClassifierConvLSTM(
            n_features=data.data.shape[1],
            n_layers=2,
            n_hidden=274,
            p_dropout_lstm=0.54,
            p_dropout_conv=0.54,
            n_outputs=n_labels,
            kernel_size=4)
        wrapper = RNNClassifierWrapper(
            model=model,
            save=f"{save_dir}/deep_curve_reverse_{reverse}_group_{i}.pth",
            op_kwargs={"lr": 0.01, "weight_decay": 0}
        )
        statistics = wrapper.fit(
            train_dataset=unit_s,
            test_dataset=cont_t,
            batch_size=16,
            epochs=epochs,
            train_fraction=train_fraction
        )
        statistics["iteration"] = i
        r_statistics += [statistics]

    return pd.concat(r_statistics)
