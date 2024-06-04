#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:00:00 2024
@author: ike
"""


import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from flypy.macros.chang.utils import ControlEncoder


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
START = -3.5
STOP = 4.0
START_N = START
STOP_N = -1
START_S = -1
STOP_S = 3


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
C_INTERVAL = "2 week"
N_DAYS = 14
# extracted column used to create trial ground truth labels
IDENTIFIER = C_IND_LABEL

FIRST_BLOCK = 9
REF_INTERVAL = 2


"""
Following parameters do not exist in pickled DataFrame. They will be extracted
from additional dimensions of C_TRACE array to reduce data dimensionality into
a (N traces x N timepoints) array.
"""
# labels of frequency ranges to unpack from electrode dimension, list[str]
F_LABEL = ["hga", "lfs"]
# column specifying electrode source of corresponding trace, str
C_ELECTRODE = "electrode"
# column specifying frequency source of corresponding trace, str
C_FREQUENCY = "frequency"


# # columns from PICKLE data to load
# CS_PICKLE = [C_TIME, C_BLOCK, IDENTIFIER, C_TRACE]
# # columns that need to be expanded such that each trial gets a unique row
# CS_EXPLODE = [IDENTIFIER, C_TRACE]
# # expanded columns
# CS_EXPAND = [C_FREQUENCY, C_ELECTRODE]
# # columns specifying relevant metadata for corresponding traces
# CS_METADATA = [C_DAY, C_BLOCK, IDENTIFIER]
# # columns by which to group traces for pairwise analysis
# CS_GROUP = [C_DAY, C_FREQUENCY, C_ELECTRODE]
# # columns by which to group traces for signal-to-noise ratio calculation
# CS_SNR = [C_DAY, IDENTIFIER, C_FREQUENCY, C_ELECTRODE]
# #
# CS_VARIABLES = [IDENTIFIER, C_DAY]
# IDX_L = 0
# IDX_G = 1


"""
encoders should define fit(X, y) and transform(X) methods.
"""
_idx_noise = np.arange(0, HZ, dtype=int)
_idx_signal = np.arange(HZ, STOP_S * HZ, dtype=int)
TOP_FEATURE = "mean."
TOP_FREQUENCIES = ["hga"]  # entry in F_LABEL or F_LABEL itself, list
FEATURES = {
    "sum": np.sum,
    "max": np.max,
    "min": np.min,
    "median:": np.median,
    "mean": np.mean,
    "s.d.": np.std,
    "a.u.c": np.trapz,
    "r.m.s.": (lambda x: np.sqrt(np.mean(x**2))),
    "s.n.r.": (lambda x: np.mean(x[_idx_signal]) / np.mean(x[_idx_noise]))
}
ENCODERS = {
    "Null": [ControlEncoder],
    "PCA": [PCA],
    "FA": [FactorAnalysis],
    "fICA": [FastICA]
    # "poly kPCA": [KernelPCA, {"kernel": "poly"}],
    # "rbf kPCA": [KernelPCA, {"kernel": "rbf"}],
    # "sigmoid kPCA": [KernelPCA, {"kernel": "sigmoid"}],
    # "cosine kPCA": [KernelPCA, {"kernel": "cosine"}]
}
CLASSIFIERS = {
    "Random Forest": [RandomForestClassifier],
    "LDA": [LinearDiscriminantAnalysis],
    "Log Reg": [LogisticRegression, {"max_iter": 2000}],
    "KNN": [KNeighborsClassifier]
}
# linalg normalization func, x is a 1d time series trace
NORM_FUNC = lambda x: x / (
    np.linalg.norm(x, ord=2) if np.linalg.norm(x, ord=2) != 0 else 1)
