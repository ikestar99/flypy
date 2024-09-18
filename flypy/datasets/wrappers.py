#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:00:00 2024
@author: ike
"""


import numpy as np
import torch

from torch.utils.data import Dataset

from flypy.datasets.timeseries import TimeSeries
from flypy.datasets.featurevector import FeatureVector


"""
This module contains convenience wrappers to use other dataset classes as input
to pytorch models. In short, each class is built to translate the data storage
architecture of the other classes in this module and translate them to a new
format with __len__ and __getitem__ methods organized according to some input
logic on the underlying metadata.
"""


class TimeVectorClassificationWrapper(Dataset):
    def __init__(
            self,
            data: FeatureVector,
            label: str,
            times: str,
            window: slice = slice(None)
    ):
        super(TimeVectorClassificationWrapper, self).__init__()
        levels = [i for i in data.meta.index.names if i not in (label, times)]

        # group timepoint vectors with identical metadata together
        meta = data.meta.copy().reset_index(drop=False).sort_values(by=[times])
        meta = meta.groupby(levels)[
            [label, data.meta.columns[-1]]].agg(lambda x: list(x))
        meta[label] = meta[label].apply(lambda x: x[0])

        # set instance attributes
        self.data = data.data
        self.meta = meta
        self.window = window

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        sample = torch.from_numpy(
            self.data[self.meta.iloc[idx, -1], self.window])
        label = self.meta.iloc[idx, 0]
        return sample, label


class TimeVectorSequenceWrapper(Dataset):
    def __init__(
            self,
            data: FeatureVector,
            times: str,
            window: int = None
    ):
        super(TimeVectorSequenceWrapper, self).__init__()
        levels = [i for i in data.meta.index.names if i != times]

        # group timepoint vectors with identical metadata together
        meta = data.meta.copy().groupby(level=levels)[
            [data.meta.columns[-1]]].agg(lambda x: list(x))

        # set instance attributes
        self.data = data.data
        self.meta = meta

        self.duration = data.data.shaoe[0] // meta.shaoe[0]
        self.window = self.duration if window is None else window
        self.repeats = self.duration - self.window + 1

    def __len__(self):
        return self.meta.shape[0] * self.repeats

    def __getitem__(self, idx):
        row_idx = idx // self.repeats
        time_idx = idx % self.repeats
        sample = self.data[
                 self.meta.iloc[row_idx, 0], time_idx:time_idx + self.window]
        return torch.from_numpy(sample[:-1]), torch.from_numpy(sample[-1])


class TraceWrapper(Dataset):
    """
    """
    def __init__(self):
        pass
