#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 7 11:41:02 2022

@author: ike
"""


from ..main import tqdm
from ..utils.csvcolumns import MAIN
from ..pipeline.liffile import loadLifFile
from ..pipeline.pipeline import *
from ..pipeline.directory import Directory


def _getDirectory(dataDirectory, rowDict):
    directory = Directory(
        dataDir=dataDirectory,
        cell=rowDict[MAIN["cel"]],
        date=rowDict[MAIN["dat"]],
        fly=rowDict[MAIN["fly"]],
        zplane=rowDict[MAIN["zpl"]],
        stimName=rowDict[MAIN["stn"]],
        stimTime=rowDict[MAIN["stt"]],
        LIFName=rowDict[MAIN["lif"]],
        ch0=rowDict[MAIN["ch0"]],
        ch1=rowDict[MAIN["ch1"]])
    return directory


def calciumImagingPreparation(dataDirectory, imagingCSVReader):
    print("Unpacking .lif files and sorting directories")
    for lifName in tqdm(imagingCSVReader.getColumn(MAIN["lif"], unique=True)):
        # unpack all images from the same .lif file simultaneously
        lif = None
        for rowDict in imagingCSVReader.filterRows({MAIN["lif"]: lifName}):
            directory = _getDirectory(dataDirectory, rowDict)
            createDirectories(directory)
            moveStimFiles(directory)
            if directory.lifFile is None:
                continue

            lif = (loadLifFile(directory.lifFile) if lif is None else lif)
            unpackLIF(lif, directory, idx=rowDict[MAIN["lnm"]] - 1)


def calciumImagingPipeline(dataDirectory, imagingCSVReader):
    print("Calcium imaging pipeline")
    for z in tqdm(imagingCSVReader.filterColumns(
            MAIN["cel"], MAIN["dat"], MAIN["fly"], MAIN["zpl"], unique=True)):
        # find all rows from same date, fly, cell, Z
        rowDicts = [rowDict for rowDict in imagingCSVReader.filterRows(z)]
        if rowDicts[0][MAIN["ref"]] is None:
            continue

        directories = [
            _getDirectory(dataDirectory, rowDict) for rowDict in rowDicts]
        alignZPlanes(directories, ref=rowDicts[0][MAIN["ref"]])
        for rowDict, directory in zip(rowDicts, directories):
            if not directory.stimFile:
                continue

            countFrames(directory)
            binWithCountedFrames(directory)
            makeMasks(directory, rowDict)
            if directory.maskFiles is None:
                continue

            measureRawResponses(directory)
            measureIndividualResponses(
                directory, rowDict[MAIN["bl0"]], rowDict[MAIN["bl1"]])
            measureAverageResponses(directory)


def responseAggregation(dataDirectory, imagingCSVReader):
    print("Aggregating responses")
    for z in tqdm(imagingCSVReader.filterColumns(
            MAIN["cel"], MAIN["stn"], unique=True)):
        directories = list()
        rowDicts = list()
        for rowDict in imagingCSVReader.filterRows(z):
            directory = _getDirectory(dataDirectory, rowDict)
            mapRFCenters(
                directory, ref=directory.channels[int(rowDict[MAIN["ref"]])])
            directories += [directory]
            rowDicts += [rowDict]

        filterMappedResponses(directories, rowDicts)


def integrateAndCorrelate(dataDirectory, imagingCSVReader):
    print("Integrating and correlating responses")
    for z in tqdm(imagingCSVReader.filterColumns(
            MAIN["cel"], MAIN["stn"], unique=True)):
        rowDicts = [r for r in imagingCSVReader.filterRows(z)]
        directories = [_getDirectory(dataDirectory, r) for r in rowDicts]
        integrateResponses(directories, rowDicts)
        correlateResponses(directories, rowDicts)


def plotAggregation(dataDirectory, imagingCSVReader):
    print("Generating plots")
    for z in tqdm(imagingCSVReader.filterColumns(
            MAIN["cel"], MAIN["stn"], unique=True)):
        directory = None
        for rowDict in imagingCSVReader.filterRows(z):
            directory = _getDirectory(dataDirectory, rowDict)
            plotAverageResponses(directory)

        if directory is not None:
            plotAggregateResponses(directory)
            plotAggregateIntegration(directory)
            plotAggregateCorrelation(directory)
