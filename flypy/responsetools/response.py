#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:32 2021

@author: katherinedelgado and Erin Barnhart
"""


import numpy as np
import pandas as pd
import scipy.stats as scs
import scipy.integrate as sci

from ..utils.csvcolumns import RESP


class Response(object):
    def __init__(
            self, name, data, regions, reporters, stimName, stimTime, stimState,
            stimType, timeStep, drivers=None, units="seconds", **kwargs):
        self.name = name
        self.data = data
        self.regions = regions
        self.reporters = reporters
        self.drivers = drivers
        self.stimName = stimName
        self.stimTime = stimTime
        self.stimState = stimState
        self.stimType = stimType
        self.timeStep = timeStep
        self.units = units
        for kwarg, item in kwargs.items():
            setattr(self, kwarg, item)


    def med(self):
        return np.median(self.fluorescence)
        # self.median.append(median)

    """identify stim switch points"""

    def stimswitch(self):
        ON_indices = list(np.where(np.diff(self.stim_state) == 1)[0] + 1)
        OFF_indices = list(np.where(np.diff(self.stim_state) == -1)[0] + 1)
        if ON_indices[0] < OFF_indices[0]:
            return ON_indices, OFF_indices
        else:
            return ON_indices, OFF_indices[1:]

    """break data into epochs"""

    def segment_responses(self, frames_before, frames_after):
        """points_before is the number of points before ON
        points_after is the number of points after OFF"""
        ON, OFF = self.stimswitch()
        r = []
        st_ind = []
        stim_type_ind = []
        for on, off in zip(ON, OFF):
            start = on - frames_before
            stop = off + frames_after
            if start > 0 and stop < len(self.F) + 1:
                r.append(self.F[start:stop])
                # print(len(self.F[start:stop]))
                st_ind.append(int(self.stim_type[on]))
        self.individual_responses = r
        self.stim_type_ind = st_ind
        return r, st_ind

    """get df/f"""

    def measure_dff(self, baseline_start, baseline_stop):
        # b = self.baseline_end
        ir = np.asarray(self.individual_responses)
        dff = []
        for i in ir:
            baseline = np.median(i[baseline_start:baseline_stop])
            dff.append((list(i - baseline) / baseline))
        self.dff = dff
        return dff

    def measure_average_dff(self, epoch_length):
        # b = self.baseline_end
        A = []
        stim_type = 1
        while stim_type <= np.max(self.stim_type_ind):
            r = []
            for dff, st in zip(self.dff, self.stim_type_ind):
                if st == stim_type:
                    r.append(dff[:epoch_length])
            R = np.asarray(r)
            A.append(list(np.average(R, axis=0)))
            stim_type = stim_type + 1
        self.average_dff = A
        return A

    def measure_stdev_dff(self):
        STDEV = []
        stim_type = 1
        while stim_type <= np.max(self.stim_type_ind):
            r = []
            for dff, st in zip(self.dff, self.stim_type_ind):
                if st == stim_type:
                    r.append(dff)
            R = np.asarray(r)
            STDEV.append(list(np.std(R, axis=0)))
            stim_type = stim_type + 1
        self.stdev_dff = A
        return STDEV

    def integrated_response(self, start_point, end_point):
        IR = []
        for aDFF in self.average_dff:
            IR.append(np.sum(aDFF[start_point:end_point]))
        return IR



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def median(self, start=0, stop=-1):
        return np.median(self.data[start:stop])

    def stimswitch(self):
        ON_indices = np.where(np.diff(self.stimState) == 1)[0] + 1
        OFF_indices = np.where(np.diff(self.stimState) == -1)[0] + 1
        if ON_indices[0] < OFF_indices[0]:
            return ON_indices, OFF_indices
        else:
            return ON_indices, OFF_indices[1:]

    def segment_responses(self, frames_before, frames_after):
        """
        points_before is the number of points before ON
        points_after is the number of points after OFF
        """
        ONidxs, OFFidxs = self.stimswitch()
        ONidxs = ONidxs - frames_before
        OFFidxs = OFFidxs + frames_after
        slices = [
            (on, off) for on, off in zip(ONidxs, OFFidxs)
            if (on >= 0 and off < len(self) - 1)]
        data = [self.data[:, s[0]:s[1]] for s in slices]
        minFrames = min([a.shape[1] for a in data])
        self.data = np.stack([a[:, :minFrames] for a in data], axis=1)
        self.stimType = np.array([self.stimType[s[0]] for s in slices])
        return self.data, self.stimType

    def measure_dff(self, baseline_start, baseline_stop):
        """get df/f"""
        baseline = np.mean(
            self.data[..., int(baseline_start):int(baseline_stop)], axis=-1)
        baseline = baseline[..., np.newaxis]
        dff = (self.data - baseline) / baseline
        return dff

    def measure_average_dff(self, epoch_length):
        # b = self.baseline_end
        A = []
        for stim_type in range(1, np.max(self.stimType) + 1, 1):
            r = []
            for dff, st in zip(self.dff, self.stim_type_ind):
                if st == stim_type:
                    r.append(dff[:epoch_length])
            R = np.asarray(r)
            A.append(list(np.average(R, axis=0)))

        self.average_dff = A
        return A

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


"============================================================================="


    def med(self):
        return np.median(self.fluorescence)
        # self.median.append(median)

    """identify stim switch points"""

    def stimswitch(self):
        ON_indices = list(np.where(np.diff(self.stim_state) == 1)[0] + 1)
        OFF_indices = list(np.where(np.diff(self.stim_state) == -1)[0] + 1)
        if ON_indices[0] < OFF_indices[0]:
            return ON_indices, OFF_indices
        else:
            return ON_indices, OFF_indices[1:]

    """break data into epochs"""

    def segment_responses(self, frames_before, frames_after):
        """points_before is the number of points before ON
        points_after is the number of points after OFF"""
        ON, OFF = self.stimswitch()
        r = []
        st_ind = []
        stim_type_ind = []
        for on, off in zip(ON, OFF):
            start = on - frames_before
            stop = off + frames_after
            if start > 0 and stop < len(self.F) + 1:
                r.append(self.F[start:stop])
                # print(len(self.F[start:stop]))
                st_ind.append(int(self.stim_type[on]))
        self.individual_responses = r
        self.stim_type_ind = st_ind
        return r, st_ind

    """get df/f"""

    def measure_dff(self, baseline_start, baseline_stop):
        # b = self.baseline_end
        ir = np.asarray(self.individual_responses)
        dff = []
        for i in ir:
            baseline = np.median(i[baseline_start:baseline_stop])
            dff.append((list(i - baseline) / baseline))
        self.dff = dff
        return dff

    def measure_average_dff(self, epoch_length):
        # b = self.baseline_end
        A = []
        stim_type = 1
        while stim_type <= np.max(self.stim_type_ind):
            r = []
            for dff, st in zip(self.dff, self.stim_type_ind):
                if st == stim_type:
                    r.append(dff[:epoch_length])
            R = np.asarray(r)
            A.append(list(np.average(R, axis=0)))
            stim_type = stim_type + 1
        self.average_dff = A
        return A

    def measure_stdev_dff(self):
        STDEV = []
        stim_type = 1
        while stim_type <= np.max(self.stim_type_ind):
            r = []
            for dff, st in zip(self.dff, self.stim_type_ind):
                if st == stim_type:
                    r.append(dff)
            R = np.asarray(r)
            STDEV.append(list(np.std(R, axis=0)))
            stim_type = stim_type + 1
        self.stdev_dff = A
        return STDEV

    def integrated_response(self, start_point, end_point):
        IR = []
        for aDFF in self.average_dff:
            IR.append(np.sum(aDFF[start_point:end_point]))
        return IR


"============================================================================="


def extract_response_objects(image_file, mask_file, stim_file, input_dict):
    """inputs are file names for aligned images, binary mask, and unprocessed stimulus file
	outputs a list of response objects"""
    # read files
    I = read_tifs(image_file)
    mask = read_tifs(mask_file)
    labels = segment_ROIs(mask)
    print('number of ROIs = ' + str(np.max(labels)))
    # process stimulus file
    stim_data, stim_data_OG, header = count_frames(stim_file)
    if (len(I)) != int(stim_data[-1][-1]):
        print("number of images does not match stimulus file")
        print('stimulus frames = ' + str(int(stim_data[-1][-1])))
        print('image frames = ' + str(len(I)))
    # stim_data = fix_dropped_frames(len(I),float(input_dict['time_interval']),stim_data,stim_data_OG,int(input_dict['gt_index']))
    # get frames, relative time, stimuulus type, and stimulus state from stim data
    fr, rt, st = parse_stim_file(stim_data,
                                 rt_index=int(input_dict['rt_index']),
                                 st_index=input_dict['st_index'])
    ss = define_stim_state(rt, float(input_dict['on_time']),
                           float(input_dict['off_time']))
    # measure fluorscence intensities in each ROI
    responses, num, labels = measure_multiple_ROIs(I, mask)
    # load response objects
    response_objects = []
    for r, n in zip(responses, num):
        ro = ResponseClassSimple.Response(F=r, stim_time=rt, stim_state=ss,
                                          ROI_num=n, stim_type=st)
        ro.sample_name = input_dict['sample_name']
        ro.reporter_name = input_dict['reporter_name']
        ro.driver_name = input_dict['driver_name']
        ro.stimulus_name = input_dict['stimulus_name']
        ro.time_interval = float(input_dict['time_interval'])
        response_objects.append(ro)

    return response_objects, stim_data, header, labels


def segment_individual_responses(response_objects, input_dict):
    for ro in response_objects:
        ro.segment_responses(int(input_dict['frames_before']),
                             int(input_dict['frames_after']))
        ro.measure_dff(int(input_dict['baseline_start']),
                       int(input_dict['baseline_stop']))


def measure_average_dff(response_objects, input_dict):
    for ro in response_objects:
        ro.measure_average_dff(int(input_dict['epoch_length']))


def save_raw_responses_csv(response_objects, filename):
    OUT = []
    for ro in response_objects:
        out = [ro.sample_name, ro.driver_name, ro.reporter_name,
               ro.stimulus_name, ro.ROI_num]
        out.extend(ro.F)
        OUT.append(out)
    output_header = ['sample_name', 'driver_name', 'reporter_name',
                     'stimulus_name', 'ROI_num']
    output_header.extend(
        list(np.arange(0, ro.time_interval * len(ro.F), ro.time_interval)))
    write_csv(OUT, output_header, filename)


def save_individual_responses_csv(response_objects, filename):
    OUT = []
    for ro in response_objects:
        for st, dff in zip(ro.stim_type_ind, ro.dff):
            out = [ro.sample_name, ro.driver_name, ro.reporter_name,
                   ro.stimulus_name, ro.ROI_num, st]
            out.extend(dff)
            OUT.append(out)
    output_header = ['sample_name', 'driver_name', 'reporter_name',
                     'stimulus_name', 'ROI_num', 'stim_type']
    output_header.extend(
        list(np.arange(0, ro.time_interval * len(dff), ro.time_interval)))
    write_csv(OUT, output_header, filename)


def save_average_responses_csv(response_objects, filename):
    OUT = []
    for ro in response_objects:
        for st, a in zip(ro.stim_type_ind, ro.average_dff):
            out = [ro.sample_name, ro.driver_name, ro.reporter_name,
                   ro.stimulus_name, ro.ROI_num, st]
            out.extend(a)
            OUT.append(out)
    output_header = ['sample_name', 'driver_name', 'reporter_name',
                     'stimulus_name', 'ROI_num', 'stim_type']
    output_header.extend(
        list(np.arange(0, ro.time_interval * len(a), ro.time_interval)))
    write_csv(OUT, output_header, filename)


def plot_raw_responses(response_objects, filename):
    for ro in response_objects:
        plot(ro.F)
        savefig(filename + '-' + str(ro.ROI_num) + '-raw.png', dpi=300,
                bbox_inches='tight')
        clf()


def plot_average_responses(response_objects, filename):
    for ro in response_objects:
        for a in ro.average_dff:
            plot(a)
        savefig(filename + '-' + str(ro.ROI_num) + '-average.png', dpi=300,
                bbox_inches='tight')
        clf()