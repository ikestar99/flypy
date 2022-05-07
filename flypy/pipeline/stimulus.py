#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


import numpy as np
import pandas as pd

from ..utils.csvreader import CSVReader
from ..utils.csvcolumns import STIM


class Stimulus(CSVReader):
    """
    Simple class to parse stimulus.csv files from live imaging sessions
    """

    def __init__(self, file, frames=None, threshold=1, voltage=2.437588):
        super().__init__()
        self.dfs = pd.read_csv(file)

        self.voltage = voltage
        self.threshold = threshold
        self.epochTime = None
        self.frames = None
        self.frameTime = None
        self.epochFrames = None
        self.frequency = None
        self.onTime = None
        self.offTime = None
        self.multiple = None
        if STIM["enm"] not in self:
            self._addEpochNumber()
        if STIM["frm"] not in self:
            self._countFrames(frames)
        if STIM["epc"] not in self:
            self._splitEpochs()

        self._extractOnTime()
        self._extractImagingTimings()
        self.multiple = (True if self.getColumn(
            STIM["enm"], unique=True).size > 1 else False)
        self.numCols = [str(x) for x in range(self.epochFrames)]
        self.xLabels = np.linspace(
            0, self.epochTime, num=self.epochFrames, endpoint=False)

    def _addEpochNumber(self):
        # add a dummy "epoch_number" column to csv files that lack one
        self.dfs.insert(0, STIM["enm"], 1)

    def _countFrames(self, finalFrame):
        """
        Identify stimulus frames corresponding to imaged frames
        """
        self.dfs = self.dfs.sort_values(
            [STIM["gbt"], STIM["rlt"]], ascending=True)
        # extract voltage signal, set values >= threshold to trigger voltage
        vs = self.getColumn(STIM["vol"])
        vs = (vs > self.threshold) * self.voltage
        # Calculate change in voltage signal for each stimulus frame
        vs[1:] = (vs[1:] - vs[:-1])
        vs[0] = 0
        # count imaging frames from the change in voltage signal
        frames = np.zeros((vs.size, 2))
        for n in range(1, len(vs) - 1, 1):
            frames[n] = frames[n - 1]
            if all((
                    vs[n] > vs[n - 1],
                    vs[n] > vs[n + 1],
                    vs[n] > self.threshold)):
                frames[n, 0] += 1
            elif all((
                    vs[n] < vs[n - 1],
                    vs[n] < vs[n + 1],
                    vs[n] < -self.threshold)):
                frames[n, 1] -= 1

        self.dfs.insert(
            self.dfs.shape[1], STIM["frm"],
            (frames[:, 0] * np.sum(frames, axis=-1)).astype(int))
        self.dfs = self.dfs.sort_values(STIM["frm"], ascending=True).groupby(
            [STIM["frm"], STIM["enm"]]).first().reset_index()
        self.dfs = self.dfs[
            (self.dfs[STIM["frm"]] >= 1) &
            (self.dfs[STIM["frm"]] <= finalFrame)]

    def _splitEpochs(self):
        rel = self.getColumn(STIM["rlt"])
        rel[1:] = rel[1:] - rel[:-1]
        rel[0] = 0
        rel = np.cumsum(
            (rel < (np.max(self.getColumn(STIM["rlt"])) * -0.5)).astype(int))
        self.dfs.insert(self.dfs.shape[1], STIM["epc"], rel)

    def _extractImagingTimings(self):
        """
        Extract frequency, time per frame, and epoch length information from
        stimulus CSV

        @return: tuple with following entries as floating-point numbers:
            0: temporal duration of each epoch in seconds
            1: number of frames in each epoch
            2: total number of frames
            3: temporal duration of each frame in seconds
            4: imaging frequency, averagenumber of frames per second
        @rtype: tuple
        """
        # extract epoch length as the maximum relative time within an epoch
        self.epochTime = int(np.ceil(np.max(self.getColumn(STIM["rlt"]))))
        self.frames = int(np.max(self.getColumn(STIM["frm"])))
        # extract time per frame as the average change in global time
        dfs = self.dfs.copy().groupby(STIM["frm"]).mean().reset_index()
        dfs = dfs[STIM["gbt"]].copy().to_numpy()
        self.frameTime = np.mean(dfs[1:] - dfs[:-1])
        self.epochFrames = int(self.epochTime // self.frameTime)
        # extract imaging frequency as the reciprocal of time per frame
        self.frequency = 1 / self.frameTime

    def _extractOnTime(self):
        if STIM["smt"] in self:
            dfs = self.dfs[[STIM["smt"], STIM["rlt"]]].copy()
            dfs = dfs[dfs[STIM["smt"]] == 1]
            self.onTime = np.min(dfs[STIM["rlt"]])
            self.offTime = np.max(dfs[STIM["rlt"]])

    def generateSplitEpochs(self):
        for e in self.getColumn(STIM["epc"], unique=True)[1:-1]:
            sub = self.dfs[self.dfs[STIM["epc"]] == e].copy()
            frames = sub[STIM["frm"]].tolist()
            frames = frames + ([frames[-1]] * (self.epochFrames - len(frames)))
            frames = frames[:self.epochFrames]
            number = sub.mode()[STIM["enm"]][0]
            yield e, number, frames

    def stimswitch(self):
        """
        identify stim switch points
        """
        stim_state = self.getColumn(STIM["rlt"])
        stim_state = (self.onTime < stim_state) * (stim_state < self.offTime)
        stim_state = stim_state.astype(int)
        ON_indices = list(np.where(np.diff(stim_state) == 1) + 1)
        OFF_indices = list(np.where(np.diff(stim_state) == -1) + 1)
        return ON_indices, OFF_indices

    def customTicks(self, bins, end=False):
        return np.linspace(0, self.epochTime, num=bins, endpoint=end)

    def binFrames(self, scalar):
        """
        Extract an ordering of frames needed to bin a corresponding image
        at a scalar multiple of the frameTime. Equivalent to resampling the
        image at a scalar^-1 multiple of the imaging frequency

        Note: "binning" a sample with a scalar of 1 is an identical operation
        to averaing between, but not within, all identical epochs

        @param dfs: stimulus CSV file stored as pandas dataframe. This
            CSV should already have imaging frames counted
        @type dfs: pandas.DataFrame
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
        width = self.frameTime * scalar
        # for every unique epoch type in "epoch_number" column
        for epoch in sorted(self.dfs[STIM["enm"]].unique().tolist()):
            #  isolate all stimulus rows corresponding to a particualr epoch
            dfn = self.dfs[self.dfs[STIM["enm"]] == epoch].copy()
            for t in np.arange(0, self.epochTime, width):
                # a bin is the relative time windown [t, t + width)
                dff = dfn[
                    (dfn[STIM["rlt"]] >= t) & (dfn[STIM["rlt"]] < (t + width))]
                if dff.empty:
                    continue

                frm = np.squeeze(
                    (dff[STIM["frm"]].to_numpy(dtype=int)) - 1).tolist()
                binFrames += ([frm] if type(frm) == list else [[frm]])
                stmFrames += [int(dff[STIM["enm"]].max())]
                if STIM["smt"] in self:
                    stmFrames[-1] = [int(dff["stim_type"].max())]

        return binFrames, stmFrames
