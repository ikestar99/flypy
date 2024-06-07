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
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
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
T0 = -3
T1 = 4.5

# relative start and stop times (s) for noise and signal window respectively
DTN0 = 0
DTN1 = -1 - T0  # Noise starts at TO, ends at -1 sec --> 2 sec elapsed
DTS0 = -1 - T0
DTSI = 3 - T0  # Signal starts at -1 --> 2 sec elapsed, ends at 6 sec elapsed


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
# first block to include in analysis
FIRST_BLOCK = 9
# Use all intervals <= this value as standard of comparison
REF_INTERVAL = 2


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


"""
encoders should define fit(X, y) and transform(X) methods.
"""
_idx_noise = np.arange(DTN0 * HZ, DTN1 * HZ, dtype=int)
_idx_signal = np.arange(DTS0 * HZ, DTSI * HZ, dtype=int)

# feature extraction for cosine similarity analysis
RAW_FEATURE = lambda x: np.sqrt(np.mean(x**2))
TOP_FREQUENCIES = ["hga"]  # entry in F_LABEL or F_LABEL itself, list

# FEATURES["feature name"] = f(trace[signal indices])
FEATURES = {
    "sum": lambda x: np.sum(x[_idx_signal]),
    "max": lambda x: np.max(x[_idx_signal]),
    "min": lambda x: np.min(x[_idx_signal]),
    "median": lambda x: np.median(x[_idx_signal]),
    "mean": lambda x: np.mean(x[_idx_signal]),
    "s.d.": lambda x: np.std(x[_idx_signal]),
    "a.u.c": lambda x: np.trapz(x[_idx_signal]),
    "r.m.s.": lambda x: np.sqrt(np.mean(x[_idx_signal]**2)),
    "s.n.r.": lambda x: np.mean(x[_idx_signal]) / np.mean(x[_idx_noise])
}

# FEATURES key to use for salience and PCA analysis
TOP_FEATURE = "r.m.s."

# ENCODERS["encoder name"] = [class] or [class, {kwargs}]
ENCODERS = {
    "Null": [ControlEncoder],
    "PCA": [PCA],
    "FA": [FactorAnalysis],
    "fICA": [FastICA]
}
DIM_ENCODE = "PCA"

# CLASSIFIERS["classifier name"] = [class] or [class, {kwargs}]
CLASSIFIERS = {
    "Random Forest": [RandomForestClassifier],
    "LDA": [LinearDiscriminantAnalysis],
    "Log Reg": [LogisticRegression],
    "KNN": [KNeighborsClassifier]
}

# linalg normalization func, x is a 1d time series trace
def NORM_FUNC(x):
    x = x / (np.linalg.norm(x, ord=2) if np.linalg.norm(x, ord=2) != 0 else 1)
    return x

# function used to aggregate traces across trials for correlation, similarity
TRIALS_FUNC = np.median
