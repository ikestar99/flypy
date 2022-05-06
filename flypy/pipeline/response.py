#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:32 2021

@author: katherinedelgado and Erin Barnhart
"""


import numpy as np
import pandas as pd
import scipy.stats as scs
import scipy.ndimage as scn
import scipy.integrate as sci

from ..utils.csvcolumns import RESP


class Response(object):
    def __init__(self, stimulus, mask=None, region=None, reporter_names=None):
        """
        self: is the instance, __init__ takes the instace 'self' and populates
        it with variables
        @param stimulus:
        @type stimulus: .stimulus.Stimulus
        """
        self.stimulus = stimulus
        if mask is not None:
            self.region = region
            self.reporter_names = reporter_names
            self.labeledMask, self.numROIs = self._segment_ROIs(mask)

        self.conCols = [
            RESP["reg"], RESP["roi"], RESP["chn"], RESP["szs"], RESP["avg"]]
        self.totCols = self.conCols + [RESP["enm"]]

    @staticmethod
    def _segment_ROIs(mask):
        """
        Label all ROIs in a given mask with a unique integer value

        @param mask: integer mask array on which to perform labeling
        @type mask: numpy.ndarray

        @return: mask array with all unique non-zero ROIs labeled with a unique
            integer

            second returned value is number of ROIs labeled
        @rtype: numpy.ndarray, int
        """
        labeledMask, regions = scn.label(mask)
        return labeledMask, regions

    @staticmethod
    def _generate_ROI_mask(mask, region):
        return (mask == region).astype(int)

    @staticmethod
    def _measureROIValues(rawstack, mask):
        """
        Compute mean pixel intensity in masked region of each image in a stack
        or hyperstack

        @param hyperstack: input array of images on which to extract mean value
            of masked region, hyperstack or otherwise. Should correspond to a
            single ROI
        @type hyperstack: numpy.ndarray
        @param mask: 2D mask of region within image from which to compute
            mean
        @type mask: numpy.ndarray

        @return: array of same shape as input array with the omission of the
            last two axes corresponding to YX dimensions, which have been
            collapsed into a single integer
        @rtype: numpy.ndarray
        """
        mask = (mask.astype(int) > 0).astype(float)
        ROISize = np.sum(mask)
        while mask.ndim < rawstack.ndim:
            mask = mask[np.newaxis]

        maskedArray = rawstack * mask
        maskedArray = np.sum(maskedArray, axis=(-2, -1)).astype(float)
        maskedArray = (maskedArray / ROISize).astype(float)
        return maskedArray

    @staticmethod
    def _subtractBackground(rawstack, background):
        """
        Compute mean pixel intensity in background region of each image in a
        stack or hyperstack and subtract each image-specific value from all
        pixels in the corresponding image

        @param rawstack: input array of images on which to perform background
            correction, hyperstack or otherwise
        @type rawstack: numpy.ndarray
        @param background: 2D mask of region within image from which to compute
            background
        @type background: numpy.ndarray

        @return: array of same shape as input array with the average background
            pixel intensity of each 2D image subtracted from all other pixels
            within the image
        @rtype: numpy.ndarray
        """
        background = (background > 0).astype(int)
        background = Response._measureROIValues(rawstack, background)
        background = background[..., np.newaxis, np.newaxis]
        rawstack = rawstack - background
        return rawstack

    def _dff(self, dfs, baseline_start, baseline_stop):
        """get df/f"""
        responses = dfs[self.stimulus.numCols].copy().to_numpy()
        baseline = np.mean(
            responses[:, int(baseline_start):int(baseline_stop)], axis=-1)
        baseline = baseline[:, np.newaxis]
        responses = (responses - baseline) / baseline
        dfs[self.stimulus.numCols] = responses
        return dfs

    def _dropFrames(self, dfs, start, stop):
        responses = dfs[self.stimulus.numCols].copy().to_numpy()
        responses[:, :start] = 0
        responses[:, stop:] = 0
        dfs[self.stimulus.numCols] = responses
        return dfs

    def measureRawResponses(self, rawstack):
        # first array identifies ROI, second specifies ROI size
        rois, sizes = np.unique(
            self.labeledMask[np.nonzero(self.labeledMask)], return_counts=True)
        # response array shape "TZC" -squeeze-> "TC" -moveaxis-> "CT"
        measuredROIs = [self._measureROIValues(
            rawstack, self._generate_ROI_mask(self.labeledMask, n))
            for n in rois]
        measuredROIs = [
            np.moveaxis(np.squeeze(r, axis=1), -1, 0) for r in measuredROIs]
        measuredROIs = np.concatenate(measuredROIs, axis=0)
        # duplicate ROI number, size, and identify indicator per channel
        rois = np.repeat(rois, len(self.reporter_names))
        sizes = np.repeat(sizes, len(self.reporter_names))
        # convert responses array into pandas dataframe
        columns = [str(x + 1) for x in range(rawstack.shape[0])]
        dfs = pd.DataFrame(data=measuredROIs, columns=columns)
        # add average raw response column to dataframe
        dfs.insert(0, RESP["reg"], self.region)
        dfs.insert(1, RESP["roi"], rois)
        dfs.insert(2, RESP["chn"], self.reporter_names * self.numROIs)
        dfs.insert(3, RESP["szs"], sizes)
        dfs.insert(4, RESP["avg"], np.mean(measuredROIs, axis=-1))
        dfm = dfs[[RESP["reg"], RESP["roi"], RESP["szs"]]].drop_duplicates()
        return dfs, dfm

    def measureIndividualResponses(self, dfr, baseline_start, baseline_stop):
        dft = None
        for item in self.stimulus.generateSplitEpochs():
            epoch, number, frames = item  # which, type, frames
            data = dfr[[str(f) for f in frames]].copy().to_numpy()
            dfs = dfr[self.conCols].copy()
            dfs[RESP["enm"]] = number
            dfs[RESP["rnm"]] = epoch
            dfs[self.stimulus.numCols] = data
            dft = (dfs if dft is None else pd.concat(
                (dft, dfs), axis=0, ignore_index=True))

        dft = self._dff(dft, baseline_start, baseline_stop)
        return dft

    def measureAverageResponses(self, dfi):
        N = dfi[RESP["rnm"]].max()
        dfs = dfi.groupby(self.totCols).mean().reset_index()
        dfs = dfs.drop(columns=[RESP["rnm"]])
        # add response number column to dataframe
        dfs.insert(5, RESP["nrs"], N)
        return dfs

    # exHead = ["size", "response_number", "epoch_label"]
    # rfHead = [
    #     "rmax", "pmax", ("x", "x_std"), ("y", "y_STD"),
    #     ("amplitude", "amplitude_std"), "mappable?"]
    #
    # RESP = dict(
    #     reg="region",
    #     roi="ROI",
    #     szs="sizes",
    #     chn="channel",
    #     avg="mean_PI",
    #     rnm="response_number",
    #     nrs="response N",
    #     enm=STIM["enm"],
    #     rmx="rmax",
    #     pmx="pmax",
    #     xst=("x", "x_std"),
    #     yst=("y", "y_STD"),
    #     ast=("amplitude", "amplitude_std"),
    #     map="mappable?")

    def mapReceptiveFieldCenters(self, dfa, channel, threshold=2):
        dfa = dfa[dfa[RESP["chn"]] == channel]
        dfa = dfa.sort_values(self.totCols, ascending=True, ignore_index=True)
        dfc = dfa[self.stimulus.numCols].copy()
        dfa = dfa[self.totCols]
        dfa[RESP["rmx"]] = dfc.max(axis=1)
        dfa[RESP["pmx"]] = dfc.idxmax(axis=1).astype(float)
        dfa[RESP["pmx"]] = np.where(
            ((dfa[RESP["enm"]] % 2) != 0),
            ((dfa[RESP["pmx"]] * 4) - 10),
            ((dfa[RESP["pmx"]] * -4) + 11))
        dfa[RESP["enm"]] = np.where((dfa[RESP["enm"]] < 3), 0, 1)
        dfc = dfa[self.conCols].copy().drop_duplicates(
            keep="first", ignore_index=True)
        for idx, key in enumerate(("xst", "yst", "ast")):
            dfc[RESP[key][0]] = dfa[dfa[RESP["enm"]] == idx].groupby(
                self.conCols).mean().reset_index()[RESP["pmx"]]
            dfc[RESP[key][1]] = dfa[dfa[RESP["enm"]] == idx].groupby(
                self.conCols).std().reset_index()[RESP["pmx"]]

        dfa = dfa.filter(self.conCols + [RESP["rmx"]])
        dfc[RESP["ast"][0]] = dfa.groupby(
            self.conCols).mean().reset_index()[RESP["rmx"]]
        dfc[RESP["ast"][1]] = dfa.groupby(
            self.conCols).std().reset_index()[RESP["rmx"]]
        stds = [RESP[key][1] for key in ("xst", "yst", "ast")]
        dfc[RESP["map"]] = np.where(dfc[stds].max(axis=1) < threshold, 1, 0)
        return dfc

    def filterMappedROIs(self, dfc, dfi):
        dfc = dfc[dfc[RESP["map"]] > 0][self.conCols[:2]].astype(str)
        dfi[self.conCols[:2]] = dfi[self.conCols[:2]].astype(str)
        dfn = (
            pd.merge(dfi, dfc, how="inner", on=self.conCols[:2])
            if not dfc.empty else None)
        dfn = (None if dfn is not None and dfn.empty else dfn)
        return dfn

    def integrateResponses(self, dfc, dfi):
        dfi = self.filterMappedROIs(dfc, dfi)
        if dfi is None:
            return None

        arr = dfi[self.stimulus.numCols]
        dfi = dfi[self.conCols[:3]]
        arr = sci.simps(arr, x=self.stimulus.xLabels, axis=1)
        dfi.insert(3, RESP["int"], arr)
        return dfi

    def correlateResponses(self, dfc, dfi):
        dfc = self.filterMappedROIs(dfc, dfi)
        if dfc is None or dfc[RESP["chn"]].unique().size == 0:
            return None

        reg, ROI, prv, ppv = list(), list(), list(), list()
        for r in dfc[RESP["reg"]].unique().flatten():
            dfs = dfc[dfc[RESP["reg"]] == r]
            for n in dfs[RESP["roi"]].unique().flatten():
                arr = dfs[dfs[RESP["roi"]] == n][self.stimulus.numCols]
                arr = arr.to_numpy()
                if arr.shape[0] != 2:
                    continue

                R, P = scs.pearsonr(arr[0], arr[1])
                reg += [r]
                ROI += [n]
                prv += [R]
                ppv += [P]

        dfc = pd.DataFrame({
            RESP["reg"]: reg, RESP["roi"]: ROI, RESP["prv"]: prv,
            RESP["ppv"]: ppv})
        dfc = (None if dfc.empty else dfc)
        return dfc


    # def average_dff(self, baseline_start, baseline_stop, epoch_length):
    #     # b = self.baseline_end
    #     ir = numpy.asarray(self.individual_responses)
    #     dff = []
    #     for i in ir:
    #         i = i[:epoch_length]  # clunky solution, need to fix
    #         baseline = numpy.median(i[baseline_start:baseline_stop])
    #         dff.append((list(i - baseline) / baseline))
    #     D = numpy.asarray(dff)
    #     self.average_dff = list(numpy.average(D, axis=0))
    #     return numpy.average(D, axis=0)
    #
    # def stdev_dff(self, baseline_start, baseline_stop):
    #     # b = self.baseline_end
    #     ir = numpy.asarray(self.individual_responses)
    #     dff = []
    #     for i in ir:
    #         baseline = numpy.median(i[baseline_start:baseline_stop])
    #         dff.append((list(i - baseline) / baseline))
    #     D = numpy.asarray(dff)
    #     return numpy.std(D, axis=0)
    #
    # def integrated_response(self, start_point, end_point):
    #     IR = numpy.sum(self.average_dff[start_point:end_point])
    #     return IR
