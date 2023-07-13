#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July  4 13:06:21 2023
@author: ike
"""


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
        groups,
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
        count = np.count_nonzero(groups == group)
        if count <= max_clusters:
            continue

        subset = data[groups == group]
        model, _ = optimize_clustering_inertia(subset, model, **kwargs)
        labels[groups == group] = model.labels_
        inertias[groups == group] = model.inertia_

    return labels, inertias


def plot_processed_data(
        data,
        group_labels,
        group_names,
        X_ticks,
        smooth_n=0
):
    groups = np.unique(group_labels)
    fig, axes = getFigAndAxes(rows=2, cols=groups.size, sharex=True)
    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
    #                     wspace=0.02, hspace=0.02)
    cbar = fig.add_axes([.91, .3, .03, .4])
    if smooth_n > 1:
        data = np.apply_along_axis(convolve, axis=-1, arr=data, window=smooth_n)

    for i, g in tqdm(enumerate(groups)):
        sub = group_labels == g
        if np.count_nonzero(sub) == 0:
            continue

        index = np.flatnonzero(sub)
        sub_data = pd.DataFrame(data[sub], index=index, columns=X_ticks)
        heatmap(
            axes[-1, i], sub_data, vmin=-5, vmax=10, center=0, cbar_ax=cbar,
            cmap="RdBu_r", xticklabels=1500, cbar_kws={'label': 'Z-score'})
        sub_data = sub_data.T
        sub_data.index.name = "Frame"
        sub_data.index = np.arange(data.shape[-1])
        sub_data.columns.name = "Cell"
        sns.lineplot(
            ax=axes[0, i], data=sub_data, errorbar=None,
            legend=False, lw=1)
        addAxisLabels(axes[-1, i], xlabel="Time (minutes)")
        addHorizontalAxTitle(axes[0, i], group_names[i])
        redrawAxis(axes[0, i], True, True)

    addAxisLabels(axes[0, 0], ylabel="Z-score")
    addAxisLabels(axes[1, 0], ylabel="Cell #")
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                        wspace=0.02, hspace=0.02)
    figure = figToImage("", fig)
    plt.close()
    return figure
