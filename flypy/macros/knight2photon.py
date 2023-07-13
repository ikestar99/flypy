#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July  4 13:06:21 2023
@author: ike
"""


import numpy as np
import pandas as pd

import matplotlib.colors as colors
from sklearn.cluster import KMeans
from tqdm import tqdm
from tslearn.clustering import TimeSeriesKMeans, KShape, KernelKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from ..utils.analysis import summarize, convolve, optimize_clustering_inertia
from ..utils.visualization import *


def threshold_activated_and_inhibited(
        data,
        start,
        stop,
        active_threshold=None,
        inhibit_threshold=None
):
    groups = np.zeros(data.shape[0])
    peaks = summarize(data, start, stop, axis=-1).flatten()
    if active_threshold is not None:
        groups[peaks >= active_threshold] = 1
    if inhibit_threshold is not None:
        groups[peaks <= inhibit_threshold] = -1

    return groups


def process_and_cluster_processed_data(
        data,
        group_labels,
        max_clusters,
        model,
        smooth_n=0,
        **kwargs
):
    labels = np.zeros(data.shape[0])
    inertias = np.zeros(data.shape[0])
    if smooth_n > 1:
        data = np.apply_along_axis(convolve, axis=-1, arr=data, window=smooth_n)

    for group in (1, -1):
        count = np.count_nonzero(group_labels == group)
        if count <= max_clusters:
            continue

        subset = data[group_labels == group]
        model, _ = optimize_clustering_inertia(subset, model, max_clusters, **kwargs)
        labels[group_labels == group] = model.labels_
        inertias[group_labels == group] = model.inertia_

    return labels, inertias


def plot_processed_data(
        data,
        group_labels,
        group_names,
        cluster_labels,
        X_ticks,
        smooth_n=0
):
    groups = np.unique(group_labels)
    fig, axes = getFigAndAxes(rows=groups.size, cols=3, sharex=True)
    if smooth_n > 1:
        data = np.apply_along_axis(convolve, axis=-1, arr=data, window=smooth_n)

    for i, g in tqdm(enumerate(groups)):
        sub = group_labels == g
        if np.count_nonzero(sub) == 0:
            continue

        index = np.flatnonzero(sub)
        sub_data = pd.DataFrame(data[sub], index=index, columns=X_ticks)
        heatmap(
            axes[i, -1], sub_data, vmin=-5, vmax=10, center=0, cbar=False,
            cmap="RdBu_r", xticklabels=1500, cbar_kws={'label': 'Z-score'})
        sub_data_raw = sub_data.T
        sub_data_raw.index.name = "Frame"
        sub_data_raw.index = np.arange(data.shape[-1])
        sub_data_raw.columns.name = "Cell"
        sns.lineplot(
            ax=axes[i, 0], data=sub_data_raw, errorbar=None, legend=False, lw=1)
        sub_data_cluster = sub_data
        sub_data_cluster.index = cluster_labels[sub]
        sub_data_cluster = sub_data_cluster.groupby(sub_data_cluster.index).mean()
        sub_data_cluster = sub_data_cluster.T
        sub_data_cluster.index.name = "Frame"
        sub_data_cluster.index = np.arange(data.shape[-1])
        sub_data_cluster.columns.name = "Cluster"
        sns.lineplot(
            ax=axes[i, 1], data=sub_data_cluster, errorbar=None, legend=False, lw=1)
        addVerticalAxTitle(axes[i, 0], group_names[i])
        redrawAxis(axes[0, i], True, True)

    im = axes[0, -1].get_images()[0]
    addAxisLabels(axes[-1, 0], xlabel="Time (minutes)")
    # addAxisLabels(axes[0, 0], ylabel="Z-score")
    # addAxisLabels(axes[1, 0], ylabel="Cell #")
    fig.colorbar(im, ax=axes.ravel().tolist())
    figure = figToImage("", fig)
    plt.close()
    return figure
