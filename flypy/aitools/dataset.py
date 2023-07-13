#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:28:00 2020


@author: ike
"""

import numpy as np
import os.path as op

import torch
import torch.utils.data as tud
import torchvision.transforms.functional as ttf

from flypy.utils.hyperstack import load_hyperstack


class Dataset(tud.Dataset):
    def __init__(
            self, hypTifs, Cout, Cin, Dout, depth, Yin, Xin, augment=True):
        self.indices = []
        self.augment = augment
        self.subSamp = (Cin, depth, Yin, Xin)
        self.Cout = Cout
        self.Dout = Dout
        self.mode = "T"
        self.weights = np.zeros(self.Cout)
        for file in [f for f in hypTifs if op.isfile(f)]:
            temp = load_hyperstack(file)
            temp = temp.reshape(-1, *temp.shape[2:])
            dims = temp.shape
            temp = np.unique(temp[:, -1], return_counts=True)
            for idx, c in enumerate(temp[0]):
                if c < self.Cout:
                    self.weights[c] += temp[1][idx]

            if not all((
                    (dims[0] >= depth), (dims[1] == Cin + 1),
                    (dims[2] >= Yin), (dims[3] >= Xin))):
                continue

            self.indices += [
                (file, t, y, x)
                for t in range(0, dims[0] - depth + 1, depth)
                for y in range(0, dims[2] - Yin + 1, Yin)
                for x in range(0, dims[3] - Xin + 1, Xin)]

        self.weights = 1 / np.maximum(self.weights, np.ones(self.Cout))
        self.weights = self.weights / np.sum(self.weights)
        self.indices = np.array(self.indices, dtype=object)

    def __len__(self):
        length = self.indices.shape[0]
        length = (length if self.mode == "P" else length // 4)
        length *= (2 if self.mode == "T" else 1)
        length *= (5 if self.augment else 1)
        return length

    def __getitem__(self, item):
        adx = (item % 5 if self.augment else 1)
        idx = (item // 5 if self.augment else item)
        if self.mode != "P":
            idx = (((idx * 2) - (idx % 2)) if self.mode == "T" else (idx * 4))
            idx += (0 if self.mode == "T" else (2 if self.mode == "V" else 3))

        file, T, Y, X = self.indices[idx]
        Cin, depth, Yin, Xin = self.subSamp
        sample = load_hyperstack(file)
        sample = sample.reshape(-1, *sample.shape[2:])
        sample = sample[T:T + depth, :, Y:Y + Yin, X:X + Xin]
        sample = (sample[0] if depth == 1 else sample)
        sample = np.swapaxes(sample, axis1=-3, axis2=0)
        sample = self._augment(sample, adx)
        sample = self._splitSample(sample)
        return sample

    def _augment(self, sample, adx):
        Yin, Xin = self.subSamp[-2:]
        sample = torch.tensor(sample)
        if (adx == 0) or (not self.augment):
            pass
        elif adx == 1:
            sample = ttf.resize(
                sample, size=[Yin * 2, Xin * 2],
                interpolation=ttf.InterpolationMode.NEAREST)
            sample = ttf.center_crop(sample, output_size=[Yin, Xin])
        elif adx == 2:
            sample = ttf.resize(
                sample, size=[Yin * 4, Xin],
                interpolation=ttf.InterpolationMode.NEAREST)
            sample = ttf.center_crop(sample, output_size=[Yin, Xin])
        elif adx == 3:
            sample = ttf.resize(
                sample, size=[Yin, Xin * 4],
                interpolation=ttf.InterpolationMode.NEAREST)
            sample = ttf.center_crop(sample,  output_size=[Yin, Xin])
        else:
            sample = ttf.rotate(
                sample, angle=135, fill=0,
                interpolation=ttf.InterpolationMode.NEAREST)

        return sample

    def _splitSample(self, sample):
        if self.mode == "P":
            sample = dict(In=sample[:self.subSamp[0]])
        else:
            sample = dict(In=sample[:-1], GT=torch.clamp(
                sample[-1], min=0, max=(self.Cout - 1)))
            if self.Dout == 2:
                sample["GT"] = sample["GT"][0]

        return sample

    def train(self):
        self.mode = "T"
        return self

    def valid(self):
        self.mode = "V"
        return self

    def test(self):
        self.mode = "Q"
        return self

    def predict(self):
        self.mode = "P"
        return self
