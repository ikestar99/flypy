#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:00:00 2024
@author: ike
"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


"""
Input variables to generate cross-grid ECOG activity graphs from a single
trial type (averaged across repetitions within a block)
"""
# number of columns in ECOG grid, tuple
G_SHAPE = (23, 11)
# number of electrodes in ECOG grid, int
N_ELECTRODES = 253
# recording frequency (Hz), start, stop times (sec) for raw trace data
HZ = 200
T0 = -3
T1 = 4.5

# downsampling factor
DOWN = 6

# relative start and stop times (s) for noise and signal window respectively
DTN0 = -3 - T0  # Noise starts at -3s --> 0s elapsed
DTN1 = -2 - T0  # Noise ends at -1s --> 2s elapsed
DTS0 = -1 - T0  # signal starts at -1s --> 2s elapsed
DTSI = 3 - T0  # signal ends at 3s --> 6s elapsed

# signal and noise intervals
NOISE = np.arange(int(DTN0 * HZ / DOWN), int(DTN1 * HZ / DOWN), dtype=int)
SIGNAL = np.arange(int (DTS0 * HZ / DOWN), int(DTSI * HZ / DOWN), dtype=int)


"""
Input variables to select columns in DataFrames containing raw and processed
data
"""
# column specifying data block, str
C_BLOCK = "blk"
# column specifying NATO code words, str
C_NATO = "txt_lab"
# column specifying ground truth label
C_LABEL = "one_hot_lab"
# column specifying trial label on the RT computer
C_IND_LABEL = "ind_lab"
# column specifying acquisition date, str
C_TIME = "timestamp"
# column specifying aligned ECOG data, str
C_TRACE = "aligned_neural"

# column specifying elapsed day in which data was collected
C_EDAY = "elapsed day"
# column specifying ground truth label, str
C_GROUND = "ground truth"
# column name specifying temporal interval used to group trials by days
C_INTERVAL = "week"
N_DAYS = 7  # number of elapsed days in each interval unit
C_BINNED = "continuous weeks"
# extracted column used to create trial ground truth labels
IDENTIFIER = C_IND_LABEL
# first block to include in analysis
FIRST_BLOCK = 9
# Use all intervals <= this value as standard of comparison
REF_INTERVAL = 1


"""
Following parameters will be extracted from additional dimensions of C_TRACE
array to reduce data dimensionality into an (N traces x N timepoints) array.
"""
# labels of frequency ranges to unpack from electrode dimension, list[str]
F_LABEL = ["hga", "lfs"]
# column specifying electrode source of corresponding trace, str
C_ELECTRODE = "electrode"
# column specifying frequency source of corresponding trace, str
C_FREQUENCY = "frequency"
# columns specifying timepoint of corresponding observation
C_TRIAL_TIME = "elapsed time (s)"





# F_LABEL or [entry in F_LABEL] that maximizes linear decoding performance
TOP_FREQUENCIES = ["hga"]

# electrodes to analyze, RT indices
SALIENT_ELECTRODES = [
    0, 9, 15, 16, 19, 21, 28, 31, 32, 33, 34, 35, 36, 37, 39, 43, 46, 47, 50,
    51, 54, 55, 65, 66, 67, 70, 74, 76, 78, 84, 89, 92, 97, 111, 118, 121, 122,
    123, 125, 127, 135, 141, 143, 146, 152, 155, 160, 161, 162, 163, 164, 165,
    167, 168, 171, 172, 176, 179, 180, 181, 184, 185, 187, 188, 190, 191, 195,
    197, 201, 205, 208, 209, 211, 212, 213, 215, 217, 220, 221, 225, 229, 231,
    233, 235, 238, 240, 243, 245
]

# NATO words
NATO_CODE_WORDS = [
    "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL",
    "INDIA", "JULIET", "KILO", "LIMA", "MIKE", "NOVEMBER", "OSCAR", "PAPA",
    "QUEBEC", "ROMEO", "SIERRA", "TANGO", "UNIFORM", "VICTOR", "WHISKEY",
    "XRAY", "YANKEE", "ZULU"
]

# FEATURES["feature name"] = f(trace[signal indices])
FEATURES = {
    "sum": lambda x: np.sum(x[SIGNAL]),
    "max": lambda x: np.max(x[SIGNAL]),
    "min": lambda x: np.min(x[SIGNAL]),
    "max_idx": lambda x: np.argmax(x[SIGNAL]),
    "min_idx": lambda x: np.argmin(x[SIGNAL]),
    "median": lambda x: np.median(x[SIGNAL]),
    "mean": lambda x: np.mean(x[SIGNAL]),
    "s.d.": lambda x: np.std(x[SIGNAL]),
    "a.u.c.": lambda x: np.trapz(x[SIGNAL]),
    "r.m.s.": lambda x: np.sqrt(np.mean(x[SIGNAL] ** 2)),
    # "s.n.r.": lambda x: np.mean(x[SIGNAL]) / np.mean(x[NOISE])
    "s.n.r.": lambda x: np.max(np.abs(x[SIGNAL])) / np.max(np.abs(x[NOISE]))
}

# FEATURES key to use for salience and PCA analysis
NOISE_FEATURE = "s.n.r."
TOP_FEATURE = "a.u.c"
TOP_MODEL = "LDA"

# CLASSIFIERS["classifier name"] = [class] or [class, {kwargs}]
CLASSIFIERS = {
    "Random Forest": [RandomForestClassifier],
    "LDA": [LinearDiscriminantAnalysis],
    "Log Reg": [LogisticRegression, {"max_iter": 3000}],
    "KNN": [KNeighborsClassifier]
}


"""
Special functions used throughout analysis
"""
# convert week labels to labels specifying continuous recording periods
def WEEK_TO_CHUNK(
        w: int
):
    out = "0"
    if w <= 7:
        out = "1 - 7"
    elif 16 <= w <= 35:
        out = "16 - 35"
    elif 50 <= w <= 57:
        out = "50 - 57"
    elif 59 <= w <= 63:
        out = "59 - 63"
    elif 68 <= w:
        out = "68 - 70"

    return out


# linalg normalization func, x is a 1d time series trace
def NORM_FUNC(
        x: np.ndarray
):
    x = x / (np.linalg.norm(x, ord=2) if np.linalg.norm(x, ord=2) != 0 else 1)
    return x


# Euclidean distance function
def EUCLIDEAN_SEP(
        _v_s: np.ndarray,
        _l_s: np.ndarray,
        *args,
        **kwargs
):
    _v_s = np.apply_along_axis(NORM_FUNC, axis=-1, arr=_v_s)
    cos = _v_s[:, np.newaxis] - _v_s[np.newaxis]
    cos = np.apply_along_axis(
        lambda x: np.sqrt(np.sum(x ** 2)), axis=-1, arr=cos)
    mask = (_l_s[:, None] == _l_s[None]).astype(int)
    np.fill_diagonal(mask, -1)
    # unique = [i for i in np.unique(_l_s) if np.sum(_l_s == i) >= 1]
    # out = np.concatenate([
    #     np.mean(cos[(_l_s == i)[:, None] * (_l_s != i)[None]])
    #     / cos[mask == i] for i in unique], axis=0)
    out = (
        [np.mean(cos[mask == 0]) - np.mean(cos[mask == 1])]
        if np.sum(mask == 0) * np.sum(mask == 1) != 0
        else [])
    return out


# cosine separability func, x is 2d (N labels, N timepoints)
# def EUCLIDEAN_SEP(x):
#     mask = np.ones((x.shape[0], x.shape[0])).astype(bool)
#     np.fill_diagonal(mask, False)
#     return np.mean(cosine_distances(x + 1)[mask])


# def EUCLIDEAN_SEP(x):
#     mask = np.ones((x.shape[0], x.shape[0])).astype(bool)
#     np.fill_diagonal(mask, False)
#     return np.mean(np.corrcoef(x)[mask])


# cosine separability
# def EUCLIDEAN_SEP(
#         _v_s: np.ndarray,
#         _l_s: np.ndarray,
#         *args,
#         **kwargs
# ):
#     _v_s = np.apply_along_axis(NORM_FUNC, axis=-1, arr=_v_s)
#     cos = cosine_similarity(_v_s + 1)
#     mask = (_l_s[:, None] == _l_s[None]).astype(int) * _l_s
#     np.fill_diagonal(mask, -1)
#     unique = [i for i in np.unique(_l_s) if np.sum(_l_s == i) >= 1]
#     out = np.concatenate([
#         cos[mask == i] - np.mean(cos[(_l_s == i)[:, None] * (_l_s != i)[None]])
#         for i in unique], axis=0)
#     # out = (
#     #     cos[mask == 1] - np.mean(cos[mask == 0])
#     #     if np.sum(mask == 0) * np.sum(mask == 1) != 0
#     #     else np.array([]))
#     return out


# correlation separability
# def EUCLIDEAN_SEP(
#         _v_s: np.ndarray,
#         _l_s: np.ndarray,
#         *args,
#         **kwargs
# ):
#     cos = np.corrcoef(_v_s)
#     mask = (_l_s[:, None] == _l_s[None]).astype(int)
#     np.fill_diagonal(mask, -1)
#     out = (
#         cos[mask == 1] - np.median(mask == 0)
#         if np.sum(mask == 0) * np.sum(mask == 1) != 0
#         else np.array([]))
#     return out


# seaborn figure-level to plot correlation values
def ANNOTATE(data, **kws):
    s, intercept, r, p, se = sp.stats.linregress(
        data["forward test accuracy"], data["reverse test accuracy"])
    ax = plt.gca()
    ax.text(
        .05, 1, 's = {:.4g} ± {:.2g} \nr = {:.2f}\np = {:.2g}'.format(
            s, se, r, p), transform=ax.transAxes)
