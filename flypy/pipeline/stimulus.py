#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


import numpy as np
import pandas as pd


AIN4 = ("AIN4", 2.437588)
THRESHOLD = 1
# column keys in stimulus.csv file
FRM = "frames"
RTM = "rel_time"
GTM = "global_time"
STP = "stim_type"
ENM = "epoch_number"


class Stimulus(object):
    """
    Simple class to parse stimulus.csv files from live imaging sessions
    """

    def __init__(self, file):
        """
        Instantiaate Stimulus object

        @param file: complete file path to stimulus.csv file
        @type file: string
        """
        # load entire csv file as 2D pandas dataframe and count frames
        self.dfs = pd.read_csv(file)
        if FRM not in self.dfs.columns:
            self.dfs = _countFrames(self.dfs)

        (self.epochTime, self.epochFrames, self.totalFrames,
         self.frameTime, self.frequency) = _extractImagingTimings(self.dfs)

    def __getitem__(self, key):
        """
        extract stim file dataframe column as numpy array

        @param key: name of column to extract
        @type key: string

        @return: array of values in specified column
        @rtype: numpy.ndarray
        """
        if key in self.dfs.columns:
            return self.dfs[[key]].to_numpy()

    def binFrames(self, scalar):
        """
        See _extractBinFrames() docstring

        @param scalar: multiple of interval at which to conduct binning
        @type scalar: float

        @return: list of imaging frames in each bin. Each index in list is a
        sublist of all frames that should be averaged to yield the index-th
        frame in the binned image. ie the following list of lists:
        [[1, 5, 8, 9]
         [2, 3, 6, 10]
         [4, 7, 11]]
        indicates that there are 3 binned frames from an imaging array with
        11 unbinned frames. The first binned image in this example includes all
        frames in bins[0] -- frames 1, 5, 8, and 9.

        Second returned entity is a second list that tracks the epoch frame
        corresponding to each bin in the previous list
        @rtype: list, list
        """
        binFrames, stmFrames = _extractBinFrames(
            self.dfs, self.epochTime, self.frameTime, scalar=scalar)
        return binFrames, stmFrames


def _countFrames(dfs):
    """
    Identify stimulus frames correspond to imaged frames

    @param dfs: stimulus CSV file stored as pandas dataframe
    @type dfs: pandas.DataFrame

    @return: stimulus CSV file with additional frames column wherein
        every CSV row corresponds to a unique imaged frame
    @rtype: pandas.DataFrame
    """
    # return input if frames already counted
    if FRM in dfs.columns:
        return dfs

    # add a dummy "epoch_number" column to csv files that lack one
    if ENM not in dfs.columns:
        dfs[ENM] = 1

    # extract voltage signal, set values >= threshold to trigger voltage
    vts = np.squeeze(dfs[AIN4[0]].to_numpy())
    vts = (vts > THRESHOLD).astype(int) * AIN4[1]
    # Calculate change in voltage signal for each stimulus frame
    vts[0] = 0
    vts[1:] = (vts[1:] - vts[:-1])
    # count imaging frames from the change in voltage signal
    frames = np.zeros((vts.size, 2))
    for n in range(1, len(vts) - 1, 1):
        frames[n] = frames[n - 1]
        if all(((vts[n] > vts[n - 1]), (vts[n] > vts[n + 1]))):
            # (vts[n] > Stimulus.threshold))):
            frames[n, 0] += 1
        elif all(((vts[n] < vts[n - 1]), (vts[n] < vts[n + 1]))):
            # (vts[n] < 0 - Stimulus.threshold))):
            frames[n, 1] -= 1

    dfs[FRM] = (
        frames[:, 0] * np.sum(frames, axis=-1)).astype(int)
    dfs = dfs.groupby([FRM, ENM]).median().reset_index().sort_values(
        FRM, ascending=True)
    dfs = dfs[dfs[FRM] >= 1]
    if dfs.shape[0] != dfs[FRM].max():
        dfs[FRM] = np.array(range(dfs.shape[0])) + 1
    return dfs


def _extractImagingTimings(dfs):
    """
    Extract frequency, time per frame, and epoch length information from
    stimulus CSV

    @param dfs: stimulus CSV file stored as pandas dataframe. This
        CSV should already have imaging frames counted
    @type dfs: pandas.DataFrame

    @return: tuple with following entries as floating-point numbers:
        0: temporal duration of each epoch in seconds
        1: number of frames in each epoch
        2: total number of frames
        3: temporal duration of each frame in seconds
        4: imaging frequency, averagenumber of frames per second
    @rtype: tuple
    """
    # extract epoch length as the maximum relative time within an epoch
    epochTime = np.max(dfs[RTM])
    totalFrames = np.max(dfs[FRM])
    # extract time per frame as the average change in global time
    dfs = dfs.groupby(FRM).mean().reset_index()
    dfs = dfs[GTM].to_numpy()
    frameTime = np.mean(dfs[1:] - dfs[:-1])
    epochFrames = epochTime // frameTime
    # extract imaging frequency as the reciprocal of time per frame
    frequency = 1 / frameTime
    return epochTime, epochFrames, totalFrames, frameTime, frequency


def _extractBinFrames(dfs, length, interval, scalar):
    """
    Extract an ordering of frames needed to bin a corresponding image
    at a scalar multiple of the frameTime. Equivalent to resampling the
    image at a scalar^-1 multiple of the imaging frequency

    Note: "binning" a sample with a scalar of 1 is an identical operation
    to averaing between, but not within, all identical epochs

    @param dfs: stimulus CSV file stored as pandas dataframe. This
        CSV should already have imaging frames counted
    @type dfs: pandas.DataFrame
    @param length: temporal duration of each epoch in seconds
    @type le                                      ngth: float
    @param interval: temporal duration of each frame in seconds
    @type interval: float
    @param scalar: multiple of interval at which to conduct binning
    @type scalar: float

    @return: list of imaging frames in each bin. Each index in list is a
    sublist of all frames that should be averaged to yield the index-th
    frame in the binned image. ie the following list of lists:
    [[1, 5, 8, 9]
     [2, 3, 6, 10]
     [4, 7, 11]]
    indicates that there are 3 binned frames from an imaging array with
    11 unbinned frames. The first binned image in this example includes all
    frames in bins[0] -- frames 1, 5, 8, and 9.

    Second returned entity is a second list that tracks the epoch frame
    corresponding to each bin in the previous list
    @rtype: list, list
    """
    binFrames = list()
    stmFrames = list()
    width = interval * scalar
    # for every unique epoch type in "epoch_number" column
    for epoch in sorted(dfs[ENM].unique().tolist()):
        #  isolate all stimulus rows corresponding to a particualr epoch
        dfn = dfs[dfs[ENM] == epoch].copy()
        for t in np.arange(0, length, width):
            # a bin is the relative time windown [t, t + width)
            dff = dfn[(dfn[RTM] >= t) & (dfn[RTM] < (t + width))]
            frm = np.squeeze((dff[FRM].to_numpy()) - 1).tolist()
            binFrames += ([frm] if type(frm) == list else [[frm]])
            stmFrames += [dff[ENM].max()]
            if STP in dff.columns:
                stmFrames[-1] = [dff["stim_type"].max()]

    return binFrames, stmFrames
