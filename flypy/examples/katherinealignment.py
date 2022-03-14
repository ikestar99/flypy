#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 7 11:41:02 2022

@author: ike
"""


from ..main import tqdm
from ..utils.pathutils import *
from ..utils.hyperstack import (
    saveTZCYXTiff, concatenateTZCYXArray, splitTZCYXArray)
from ..pipeline.liffile import loadLifFile, getLifImage
from ..pipeline.alignment import alignStack


def alignImagesInLIFFile(lifPath, indexes, channel):
    lifFile = loadLifFile(lifPath)
    arrays = [getLifImage(lifFile, idx=idx - 1) for idx in indexes]
    arrays, lengths = concatenateTZCYXArray(arrays, axis=0)
    for x in tqdm(range(3)):
        arrays = alignStack(arrays, channel, mode="rigid")

    arrays = splitTZCYXArray(arrays, lengths, axis=0)
    saveDir = getPath(changeExt(lifPath))
    makeDir(saveDir)
    for i, array in enumerate(arrays):
        savePath = getPath(
            saveDir, "LIF Image {}".format(str(indexes[i])), ext="tif")
        saveTZCYXTiff(savePath, array, "TZCYX")