#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 7 11:41:02 2022

@author: ike
"""


import numpy as np
import pandas as pd

from ..utils.filepath import Filepath
from ..utils.pipeutils import smooth
from ..utils.hyperstack import (
    saveTZCYXTiff, loadTZCYXTiff, savePillowArray, binStack)
from ..utils.visualization import lineGraph, barPlot
from ..utils.csvcolumns import RESP, MAIN
from ..pipeline.response import Response
from ..pipeline.liffile import getLifImage
from ..pipeline.stimulus import Stimulus
from ..pipeline.alignment import alignStack


def createDirectories(directory):
    directory.sampDir.make()
    directory.imgDir.make()
    directory.maskDir.make()
    directory.measDir.make()
    directory.figDir.make()
    directory.totMeasDir.make()
    directory.totFigDir.make()


def unpackLIF(lifFile, directory, idx):
    # load imaging settings CSV file
    """
    following code block in for loop will unpack lif files and form appropriate
    directory structure for all samples to be analyzed, skipping any samples
    for which the procedure is already complete
    """
    if directory.imgFile or not directory.lifFile:
        return

    # extract image from lifFile object and save
    hyperstack = getLifImage(lifFile, idx=idx)
    saveTZCYXTiff(directory.imgFile, hyperstack, shape="TZCYX")


def moveStimFiles(directory):
    if not directory.stimFile:
        return

    # move associated stim file to imageDirectory
    newStim = Filepath(directory.imgDir, directory.stimFile[-1], ext="csv")
    if not newStim:
        directory.stimFile.move(newStim)


def alignZPlanes(directories, ref, n=3):
    """
    following code block in for loop will iterate over unpacked
    samples and align all images acquired from the same z-plane
    """
    hyperstacks = [bool(d.imgFile) for d in directories]
    if directories[0].projFile or not all(hyperstacks):
        return

    # load each image as hyperstack
    hyperstacks = [loadTZCYXTiff(d.imgFile) for d in directories]
    # record number of frames in each hyperstack
    frames = [h.shape[0] for h in hyperstacks]
    # concatenate all images in sample for cross-stimulus alignment
    hyperstack = (
        np.concatenate(hyperstacks, axis=0) if len(hyperstacks) > 1
        else hyperstacks[0])
    # align hyperstack three times
    for _ in range(n):
        hyperstack = alignStack(hyperstack, channel=int(ref), mode="rigid")

    # split alighned hyperstacks into original rows and save each one
    frame = 0
    for x, d in enumerate(directories):
        aligned = hyperstack[frame:frame + frames[x]]
        saveTZCYXTiff(d.imgFile, aligned, "TZCYX")
        frame += frames[x]

    # create and save average projection of aligned hyperstack
    hyperstack = np.mean(hyperstack, axis=0, keepdims=True)
    saveTZCYXTiff(directories[0].projFile, hyperstack, "TZCYX")


def countFrames(directory):
    """
    following code block in for loop will iterate over unpacked
    samples and count frames + bin all aligned images
    """
    # if images already binned, skip current sample
    if not directory.stimFile or directory.frameFile:
        return

    # load stack associated with current row
    hyperstack = loadTZCYXTiff(directory.imgFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.stimFile, frames=hyperstack.shape[0])
    stimulus.save(directory.frameFile)


def binWithCountedFrames(directory, scalar=2):
    """
    following code block in for loop will iterate over unpacked
    samples and count frames + bin all aligned images
    """
    # if images already binned, skip current sample
    if not directory.frameFile or directory.binFile:
        return

    # load stack associated with current row
    hyperstack = loadTZCYXTiff(directory.imgFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.frameFile, frames=hyperstack.shape[0])
    # extract list if images/bin from stimulus object
    binList, _ = stimulus.binFrames(scalar)
    # bin, and save stack
    hyperstack = binStack(hyperstack, binList)
    saveTZCYXTiff(directory.binFile, hyperstack, "TZCYX")


def makeMasks(directory, rowDict):
    """
    following code block in for loop will iterate over unpacked samples and
    generate masks for each stimulus-paired sample using machine learning
    well, it will in theory. At some point this will work... ¯\_(ツ)_/¯
    """
    # mask = loadTZCYXTiff(directory.projFile)[0, 0, -1]
    # maskFile = Filepath(directory.maskDir, rowDict[MAIN["lay"]], ext="tif")
    # if not maskFile:
    #     saveTZCYXTiff(maskFile, mask, shape="YX")
    pass


def measureRawResponses(directory):
    """
    following code block in for loop will iterate over unpacked samples and
    measure raw responses per ROI in each specified mask
    """
    # if images already binned, skip current sample
    if not directory.imgFile or not directory.frameFile or directory.rawFile:
        return

    # load stack associated with current row
    hyperstack = loadTZCYXTiff(directory.imgFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.frameFile)
    dfrt = None
    dfmt = None
    for maskFile in directory.maskFiles:
        mask = loadTZCYXTiff(maskFile)[0, 0, 0]
        response = Response(
            stimulus=stimulus, mask=mask,
            region=Filepath.stabilize(maskFile[-1]),
            reporter_names=directory.channels)
        saveTZCYXTiff(maskFile, response.labeledMask, shape="YX")
        dfr, dfm = response.measureRawResponses(hyperstack)
        dfrt = (dfr if dfrt is None else pd.concat(
            (dfrt, dfr), axis=0, ignore_index=True))
        dfmt = (dfm if dfmt is None else pd.concat(
            (dfmt, dfm), axis=0, ignore_index=True))

    dfrt.to_csv(directory.rawFile, encoding="utf-8", index=False)
    dfmt.to_csv(directory.measFile, encoding="utf-8", index=False)


def measureIndividualResponses(directory, baseline_start, baseline_stop):
    """
    following code block in for loop will iterate over unpacked samples and
    measure raw responses per ROI in each specified mask
    """
    # if images already binned, skip current sample
    if not directory.rawFile or not directory.frameFile or directory.indFile:
        return

    # load stack associated with current row
    dfrt = pd.read_csv(directory.rawFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.frameFile)
    response = Response(stimulus)
    dfit = response.measureIndividualResponses(
        dfrt, baseline_start, baseline_stop)
    dfit.to_csv(directory.indFile, encoding="utf-8", index=False)


def measureAverageResponses(directory):
    """
    following code block in for loop will iterate over unpacked samples and
    measure raw responses per ROI in each specified mask
    """
    # if images already binned, skip current sample
    if not directory.indFile or not directory.frameFile or directory.avgFile:
        return

    # load stack associated with current row
    dfit = pd.read_csv(directory.indFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.frameFile)
    response = Response(stimulus=stimulus)
    dfat = response.measureAverageResponses(dfit)
    dfat.to_csv(directory.avgFile, encoding="utf-8", index=False)


def mapRFCenters(directory, ref):
    if not directory.frameFile:
        return

    stimulus = Stimulus(directory.frameFile)
    response = Response(stimulus)
    if not stimulus.multiple or directory.rfcFile:
        return

    dfa = pd.read_csv(directory.avgFile)
    dfr = response.mapReceptiveFieldCenters(dfa, ref)
    dfr.to_csv(directory.rfcFile, encoding="utf-8", index=False)


def filterMappedResponses(directories, rowDicts):
    dfnt = None
    for directory, rowDict in zip(directories, rowDicts):
        if not directory.frameFile or not directory.rfcFile:
            continue

        stimulus = Stimulus(directory.frameFile)
        response = Response(stimulus)
        if stimulus.multiple:
            continue

        dfc = pd.read_csv(
            directory.rfcFile, usecols=(response.conCols[:3] + [RESP["map"]]))
        dfi = pd.read_csv(directory.indFile)
        dfn = response.filterMappedROIs(dfc, dfi)
        if dfn is None:
            continue

        dfn.insert(0, MAIN["dat"], rowDict[MAIN["dat"]])
        dfn.insert(1, MAIN["fly"], rowDict[MAIN["fly"]])
        dfn.insert(2, MAIN["zpl"], rowDict[MAIN["zpl"]])
        dfnt = (
            dfn if dfnt is None else pd.concat(
                (dfnt, dfn), axis=0, ignore_index=True))

    if dfnt is not None:
        dfnt.to_csv(directories[0].totMeasFile, encoding="utf-8", index=False)


def integrateResponses(directories, rowDicts):
    dfit = None
    for directory, rowDict in zip(directories, rowDicts):
        if not directory.frameFile or not directory.rfcFile:
            continue

        stimulus = Stimulus(directory.frameFile)
        response = Response(stimulus)
        if stimulus.multiple:
            continue

        dfc = pd.read_csv(
            directory.rfcFile, usecols=(response.conCols[:3] + [RESP["map"]]))
        dfi = pd.read_csv(directory.indFile)
        dfi = response.integrateResponses(dfc, dfi)
        if dfi is None:
            continue

        dfi.insert(0, MAIN["dat"], rowDict[MAIN["dat"]])
        dfi.insert(1, MAIN["fly"], rowDict[MAIN["fly"]])
        dfi.insert(2, MAIN["zpl"], rowDict[MAIN["zpl"]])
        dfit = (
            dfi if dfit is None else pd.concat(
                (dfit, dfi), axis=0, ignore_index=True))

    if dfit is not None:
        dfit.to_csv(directories[0].totIntFile, encoding="utf-8", index=False)


def correlateResponses(directories, rowDicts):
    dfct = None
    for directory, rowDict in zip(directories, rowDicts):
        if not directory.frameFile or not directory.rfcFile:
            continue

        stimulus = Stimulus(directory.frameFile)
        response = Response(stimulus)
        if stimulus.multiple:
            continue

        dfc = pd.read_csv(
            directory.rfcFile, usecols=(response.conCols[:3] + [RESP["map"]]))
        dfa = pd.read_csv(directory.avgFile)
        dfi = response.correlateResponses(dfc, dfa)
        if dfi is None:
            continue

        dfi.insert(0, MAIN["dat"], rowDict[MAIN["dat"]])
        dfi.insert(1, MAIN["fly"], rowDict[MAIN["fly"]])
        dfi.insert(2, MAIN["zpl"], rowDict[MAIN["zpl"]])
        dfct = (
            dfi if dfct is None else pd.concat(
                (dfct, dfi), axis=0, ignore_index=True))

    if dfct is not None:
        dfct.to_csv(directories[0].totCorFile, encoding="utf-8", index=False)


def plotAverageResponses(directory):
    """
    code block equivalent to former "plot average responses" portion of
    automated endopy pipeline
    """
    if not directory.indFile or not directory.frameFile or directory.avgFig:
        return

    dfit = pd.read_csv(directory.indFile)
    stimulus = Stimulus(directory.frameFile)
    channels = dfit[RESP["chn"]].unique().tolist()
    dfit[RESP["roi"]] = (
        dfit[RESP["reg"]].astype(str) + " ROI " +
        dfit[RESP["roi"]].astype(int).astype(str))
    ROIs = dfit[RESP["roi"]].unique().tolist()
    dfit[RESP["reg"]] = (
        dfit[RESP["roi"]] + " " +
        dfit[RESP["chn"]].astype(str) + " epoch " +
        dfit[RESP["enm"]].astype(int).astype(str))
    dfit = dfit.drop(columns=[
        RESP["roi"], RESP["szs"], RESP["avg"], RESP["rnm"], RESP["chn"],
        RESP["enm"]])
    dfat = dfit.groupby([RESP["reg"]]).mean().reset_index()
    dfst = dfit.groupby([RESP["reg"]]).sem().reset_index()
    dfat = dfat.set_index(RESP["reg"]).rename_axis(None, axis=0)
    dfat = dfat.T.set_index(stimulus.xLabels)
    dfst = dfst.set_index(RESP["reg"]).rename_axis(None, axis=0)
    dfst = dfst.T.set_index(stimulus.xLabels)
    figures = []
    columns = dfat.columns.values.tolist()
    for r in ROIs:
        subset = [c for c in columns if r in c]
        dfa = dfat[subset].copy()
        dfs = dfst[subset].copy()
        figures += [lineGraph(
            Ys=dfa, dYs=dfs, titles=(r, "Time (s)", "ΔF/F"),
            light=(stimulus.onTime, stimulus.offTime), subs=channels)]

    savePillowArray(str(directory.avgFig), figures)


def plotAggregateResponses(directory):
    if not directory.totMeasFile:
        return

    dfit = pd.read_csv(directory.totMeasFile)
    stimulus = Stimulus(directory.frameFile)
    if stimulus.multiple:
        return

    dfit = dfit[[RESP["reg"], RESP["chn"]] + stimulus.numCols]
    figures = []
    for r in dfit[RESP["reg"]].unique().tolist():
        dfr = dfit[dfit[RESP["reg"]] == r].copy()
        dfr = dfr.drop(columns=[RESP["reg"]])
        channels = dfr[RESP["chn"]].unique().tolist()
        dfa = dfr.copy().groupby([RESP["chn"]]).mean().reset_index()
        dfs = dfr.copy().groupby([RESP["chn"]]).sem().reset_index()
        dfa = dfa.set_index(RESP["chn"]).rename_axis(None, axis=0)
        dfa = dfa.T.set_index(stimulus.xLabels)
        dfs = dfs.set_index(RESP["chn"]).rename_axis(None, axis=0)
        dfs = dfs.T.set_index(stimulus.xLabels)
        for column in dfa.columns.values.tolist():
            dfa[column] = smooth(dfa[[column]].to_numpy().flatten())
        figures += [lineGraph(
            Ys=dfa, dYs=dfs, titles=(r, "Time (s)", "ΔF/F"),
            light=(stimulus.onTime, stimulus.offTime), subs=channels)]

    savePillowArray(str(directory.totAvgFig), figures)


def plotAggregateIntegration(directory):
    if not directory.totIntFile:
        return

    data = pd.read_csv(directory.totIntFile, usecols=[
        RESP["reg"], RESP["chn"], RESP["int"]])
    figure = barPlot(
        data, catCol=RESP["reg"], valCol=RESP["int"], hueCol=RESP["chn"],
        title="Integrated Responses", xLabel="area under response curve (s)")
    savePillowArray(str(directory.totIntFig), [figure])


def plotAggregateCorrelation(directory):
    if not directory.totCorFile:
        return

    data = pd.read_csv(directory.totCorFile, usecols=[
        RESP["reg"], RESP["prv"], RESP["ppv"]])
    data = data.melt(
        id_vars=[RESP["reg"]], value_vars=[RESP["prv"], RESP["ppv"]],
        var_name="statistic", value_name="value")
    figure = barPlot(
        data, catCol=RESP["reg"], valCol="value", rowCol="statistic",
        title="Correlated Responses")
    savePillowArray(str(directory.totCorFig), [figure])
