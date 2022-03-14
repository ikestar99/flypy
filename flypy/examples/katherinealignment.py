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
    """
    Example of complex function to align all images acquired from same z-plane
    within a .lif file by tileing simpler hyperstack and alignment functions

    @param lifPath: filepath to .lif file that contains images to be aligned
    @type lifPath: string
    @param indexes: indexes of images to be aligned within .lif file. Note,
        .lif files are NOT 0-indexed so the first image in your .lif file has
        an index of 1. ie, indexes= [5, 6] means that the code will group and
        align the FIFTH and SIXTH images in your .lif file, regardless of the
        number that appears next to these jobs in leica or on your imaging
        sheet. All images to be aligned together MUST have the shape shape
        (same number of channels, same X/Y dimensions, same z planes) except
        in the alignment axis, which is the time dimension. This parameter is
        a list -- if you want to align a single index within your .lif file,
        input that index as indexes=[index] rather than indexes=index
    @type indexes: list
    @param channel: reference channel on which to conduct transformation. Note:
        unlike the indexes paramterer, leica .lif channels ARE 0-indexed. So,
        channels=0 indicates an alighnment based on CH0 in leica. To align
        based on RGECO in ERXXX/RGECO imaging experiments, this parameter
        should be 1
    @type channel: int
    """
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