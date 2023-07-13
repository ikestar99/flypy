#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 03:13:56 2021

@author: ike
"""

import numpy as np

from flypy.utils.pathutils import get_path
from flypy.utils.analysis import z_score
from flypy.macros.knight2photon import (
    threshold_activated_and_inhibited, process_and_cluster_processed_data,
    plot_processed_data)


def knight_lab_two_photon_analysis():
    directory = "/Users/ike/Desktop/test"
    frequency = 5
    b_start = 1
    b_stop = 15
    stim_start = 15
    stim_stop = 25
    act_thresh = 1
    inhib_thresh = -0.5
    label_names = ["Inhibited Cells", "Nada", "Activated Cells"]

    total_traces = np.load(get_path(directory, "F", ext="npy"))
    neuropil_traces = np.load(get_path(directory, "Fneu", ext="npy"))
    cells = np.load(get_path(directory, "iscell", ext="npy"))[..., 0] >= 1

    data = (total_traces - (0.7 * neuropil_traces))[cells]
    data = z_score(
        array=data,
        start=b_start * 60 * frequency,
        stop=b_stop * 60 * frequency,
        axis=-1)
    group_labels = threshold_activated_and_inhibited(
        data=data,
        start=stim_start * 60 * frequency,
        stop=stim_stop * 60 * frequency,
        active_threshold=act_thresh,
        inhibit_threshold=inhib_thresh)
    # clustering_labels, clustering_inertias = process_and_cluster_raw_data(
    #     data=data,
    #     groups=group_labels,
    #     max_clusters=10,
    #     model="timeserieskmeans",
    #     smooth_n=5,
    #     n_clusters=2,
    #     max_iter=10,
    #     n_init=2,
    #     random_state=0)
    times = np.arange(data.shape[-1]) / (60 * frequency)
    figure = plot_processed_data(data, group_labels, label_names, times, 50)
    test_save = "/Users/ike/Desktop/test/test_image.tif"
    figure.save(test_save, compression="tiff_deflate")


def __main__():
    knight_lab_two_photon_analysis()


if __name__ == "__main__":
    __main__()
