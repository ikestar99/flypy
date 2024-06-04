#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 03:13:56 2021

@author: ike
"""

import numpy as np
import pickle

from flypy.datasets.timeseries import TimeSeries


def temp_block():
    block = 0
    # date = "2022_09_16-11_01_40"
    label = 0
    n_rows = 23
    n_cols = 11
    electrodes = 253
    hz = 200
    start_time = -3.5
    stop_time = 4
    block_col = "blk"
    name_col = "txt_lab"
    day_col = "timestamp"
    data_col = "aligned_neural"
    label_col = "one_hot_lab"
    high_name = "hga"
    low_name = "low"
    time_col = "Time (ms)"
    grid_col = "electrode"
    range_col = "frequency range"

    aligned_data_path = "/Users/ike/Documents/Lab/Chang Lab/Data/df_ike.pkl"
    save_figure_path = "/Users/ike/Desktop/img1.png"

    time_labels = np.arange(start_time, stop_time, 1 / hz)
    include_cols = [block_col, name_col, label_col, data_col]
    expand_cols = [name_col, label_col, data_col]

    with open(aligned_data_path, "rb") as f:
        data = pickle.load(f).head(n=2)


def __main__():
    test_data = np.arange(72).reshape(4, 6, 3)
    # test_timepoints = np.arange(0, 3, 0.5)
    # expand = [(-1, "electrode", [1, 2, 3])]
    # block = 0
    # timestamp = ["Mon", "Mon", "Tues", "Tues"]
    #
    # label = ["a", "b", "c", "d"]
    # temp = TimeSeriesDataSet(test_data, test_timepoints, expand=expand, time=timestamp, label=label)
    # temp = temp[{"time": ["Mon"]}]
    #
    # temp = temp + temp
    # print(temp.get_subgroup_counts(["time"]))
    print(test_data.shape)

    # print(np.transpose(test_data, (0, -1, 1)).reshape(-1, 6))
    # maxs = 7
    # dim_order = list(range(maxs))
    # newa = dim_order.copy()
    # for i, d in enumerate([2, 5, -1]):
    #     dim_order.insert(i + 1, dim_order.pop(d))
    #     print(i+1, dim_order)
    #
    #     print("\n")


if __name__ == "__main__":
    __main__()
