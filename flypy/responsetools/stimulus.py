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


voltage=2.437588


def count_frames(
        filename, threshold=1, volCol="AIN4", fCol="frames",
        gtCol="global_time", dtCol="dt"):
    """
    Reads in a stimulus output file and assigns an image frame number to each
    stimulus frame
    """
    stim = CSVReader.fromFile(filename)
    # R = np.asarray(rows, dtype='float') # convert stim file list to an array
    # output_array=np.zeros((R.shape[0],R.shape[1]+2))
    # header.extend(['dt','frames'])

    vs = stim.getColumn(volCol)
    vs[1:] = (vs[1:] - vs[:-1])
    vs[0] = 0
    # count image frames based on the change in voltage signal
    count_on, count_off = 0, 0
    frame_labels = [0]
    # F_on = [0]; F_off = [0]
    for n in range(1, len(vs) - 1, 1):
        if all((
                vs[n] > vs[n - 1],
                vs[n] > vs[n + 1],
                vs[n] > threshold)):
            count_on += 1
        elif all((
                vs[n] < vs[n - 1],
                vs[n] < vs[n + 1],
                vs[n] < -threshold)):
            count_off -= 1

        # F_on.extend([count_on]); F_off.extend([count_off])
        frame_labels += [count_on * (count_on + count_off)]

    stim = stim.setColumn(fCol, frame_labels + [0])
    stim = stim.sortColumn(fCol)
    stim = stim.thresholdColumn(fCol, 1, ">=")
    stim = stim.dropDuplicates(fCol)
    gt = stim.getColumn(gtCol)
    gt[0] = 0
    stim = stim.setColumn(dtCol, gt)
    return stim


def find_dropped_frames(
        frames, time_interval, oldStim, newStim, fCol="frames",
        gtCol="global_time", dtCol="dt", ):
    sFrames = newStim[-1][fCol]
    print("N image frames: {:d} \nN stim frames: {:d}".format(frames, sFrames))
    if sFrames == frames:
        print("looks fine!")
        return

    print("uh oh!")
    target_T = frames * time_interval
    stim_T = np.sum(newStim.getColumn(dtCol))
    print("total time should be {:.3f}s, got {:.3f}s ".format(
        target_T, stim_T))
    max_t_step = np.max(newStim.getColumn(dtCol))
    if np.round(max_t_step / time_interval) < 2:
        print(
            ("stim frames and image frames do not match, but no dropped "
             "frames found... double check the stim file :-( "))
        return

    print("stimulus dropped at least one frame!")
    OUT = []
    num_df = 0
    for rowDict in newStim:
        if np.round(rowDict[dtCol] / time_interval) >= 2:
            num_df = num_df + 1
            gt_dropped = rowDict[gtCol] - time_interval
            stim_frame = np.searchsorted(
                oldStim.getColumn(gtCol), gt_dropped)
            print("check row {} of original stim file (maybe)".format(
                stim_frame))

        OUT.append(rowDict)

    print("found {} potential dropped frames".format(num_df))


def parse_stim_file(
        newStim, fCol="frames", rtCol="rel_time", stCol="stim_type"):
    """
    Get frame numbers, global time, relative time per epoch, and stim_state
    (if it's in the stim_file)
    """
    frames = newStim.getColumn(fCol)
    rel_time = newStim.getColumn(rtCol)
    stim_type = (
        newStim.getCol(stCol) if stCol in newStim else np.ones(frames.size))
    return frames, rel_time, stim_type


def splitEpochs(newStim, rtCol="rel_time"):
    rel = newStim.getColumn(rtCol)
    maximum = np.max(rel)
    rel[1:] = rel[1:] - rel[:-1]
    rel[0] = 0
    rel = np.cumsum((rel < (maximum * -0.5)).astype(int))
    return rel


def define_stim_state(newStim, on_time, off_time, rtCol="rel_time"):
    """
    Define stimulus state (1 = ON; 0 = OFF) based on relative stimulus time
    """
    rel_time = newStim.getColumn(rtCol)
    stim_state = ((rel_time > on_time) * (rel_time < off_time)).astype(int)
    return stim_state


def stimswitch(newStim, on_time, off_time, rtCol="rel_time"):
    """
    identify stim switch points
    """
    rel_time = newStim.getColumn(rtCol)
    stim_state = (on_time < rel_time) * (rel_time < off_time)
    stim_state = stim_state.astype(int)
    ON_indices = list(np.where(np.diff(stim_state) == 1) + 1)
    OFF_indices = list(np.where(np.diff(stim_state) == -1) + 1)
    return ON_indices, OFF_indices


def _addEpochNumber(self):
    # add a dummy "epoch_number" column to csv files that lack one
    self.dfs.insert(0, STIM["enm"], 1)


def _extractImagingTimings(
        newStim, fCol="frames", gtCol="global_time", rtCol="rel_time"):
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
    epochTime = int(np.ceil(np.max(newStim.getColumn(rtCol))))
    frames = int(np.max(newStim.getColumn(fCol)))
    # extract time per frame as the average change in global time
    dfs = newStim.dfs.copy().groupby(fCol).mean().reset_index()
    dfs = dfs[gtCol].copy().to_numpy()
    frameTime = np.mean(dfs[1:] - dfs[:-1])
    epochFrames = int(epochTime // frameTime)
    # extract imaging frequency as the reciprocal of time per frame
    frequency = 1 / frameTime
    return epochTime, epochFrames, frames, frameTime, frequency


def generateSplitEpochs(self):
    for e in self.getColumn(STIM["epc"], unique=True)[1:-1]:
        sub = self.dfs[self.dfs[STIM["epc"]] == e].copy()
        frames = sub[STIM["frm"]].tolist()
        frames = frames + ([frames[-1]] * (self.epochFrames - len(frames)))
        frames = frames[:self.epochFrames]
        number = sub.mode()[STIM["enm"]][0]
        yield e, number, frames


# def binFrames(self, scalar):
#     """
#     Extract an ordering of frames needed to bin a corresponding image
#     at a scalar multiple of the frameTime. Equivalent to resampling the
#     image at a scalar^-1 multiple of the imaging frequency
#
#     Note: "binning" a sample with a scalar of 1 is an identical operation
#     to averaing between, but not within, all identical epochs
#
#     @param dfs: stimulus CSV file stored as pandas dataframe. This
#         CSV should already have imaging frames counted
#     @type dfs: pandas.DataFrame
#     @param scalar: multiple of interval at which to conduct binning
#     @type scalar: float
#
#     @return: list of imaging frames in each bin. Each index in list is a
#     sublist of all frames that should be averaged to yield the index-th
#     frame in the binned image. ie the following list of lists:
#     [[1, 5, 8, 9]
#      [2, 3, 6, 10]
#      [4, 7, 11]]
#     indicates that there are 3 binned frames from an imaging array with
#     11 unbinned frames. The first binned image in this example includes all
#     frames in bins[0] -- frames 1, 5, 8, and 9.
#
#     Second returned entity is a second list that tracks the epoch frame
#     corresponding to each bin in the previous list
#     @rtype: list, list
#     """
#     binFrames = list()
#     stmFrames = list()
#     width = self.frameTime * scalar
#     # for every unique epoch type in "epoch_number" column
#     for epoch in sorted(self.dfs[STIM["enm"]].unique().tolist()):
#         #  isolate all stimulus rows corresponding to a particualr epoch
#         dfn = self.dfs[self.dfs[STIM["enm"]] == epoch].copy()
#         for t in np.arange(0, self.epochTime, width):
#             # a bin is the relative time windown [t, t + width)
#             dff = dfn[
#                 (dfn[STIM["rlt"]] >= t) & (dfn[STIM["rlt"]] < (t + width))]
#             if dff.empty:
#                 continue
#
#             frm = np.squeeze(
#                 (dff[STIM["frm"]].to_numpy(dtype=int)) - 1).tolist()
#             binFrames += ([frm] if type(frm) == list else [[frm]])
#             stmFrames += [int(dff[STIM["enm"]].max())]
#             if STIM["smt"] in self:
#                 stmFrames[-1] = [int(dff["stim_type"].max())]
#
#     return binFrames, stmFrames
