#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:09:32 2021

@author: katherinedelgado and Erin Barnhart
"""
import numpy as np


class Response(object):
    """class attributes"""

    """instance attributes"""
    _instance_data = {'sample_name': None,
                      'ROI_num': None,
                      'reporter_name': None,
                      'driver_name': None,
                      'F': [],
                      'stimulus_name': None,
                      'stim_time': [],
                      'stim_state': [],
                      'stim_type': [],
                      'time_step': None,
                      'units': 'seconds'}

    def __init__(self, **kws):
        for attr, value in self._instance_data.items():
            if attr in kws:
                value = kws[attr]
                setattr(self, attr, value)
            else:
                setattr(self, attr, value)

    """instance methods"""
    """find median fluorescence over time"""

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
