#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 7 11:41:02 2022

@author: ike
"""

import pandas as pd

from flypy.pipeline.filepath import Filepath
from ..utils.hyperstack import (
    save_hyperstack, load_hyperstack, _save_pil_array, binStack)
from ..utils.visualization import *
from flypy.pipeline.csvcolumns import RESP, MAIN
from flypy.pipeline.response import Response
from flypy.utils.liffile import getLifImage
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
    save_hyperstack(directory.imgFile, hyperstack, shape="TZCYX")


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
    hyperstacks = [load_hyperstack(d.imgFile) for d in directories]
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
        save_hyperstack(d.imgFile, aligned, "TZCYX")
        frame += frames[x]

    # create and save average projection of aligned hyperstack
    hyperstack = np.mean(hyperstack, axis=0, keepdims=True)
    save_hyperstack(directories[0].projFile, hyperstack, "TZCYX")


def countFrames(directory):
    """
    following code block in for loop will iterate over unpacked
    samples and count frames + bin all aligned images
    """
    # if images already binned, skip current sample
    if not directory.stimFile or directory.frameFile:
        return

    # load stack associated with current row
    hyperstack = load_hyperstack(directory.imgFile)
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
    hyperstack = load_hyperstack(directory.imgFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.frameFile, frames=hyperstack.shape[0])
    # extract list if images/bin from stimulus object
    binList, _ = stimulus.binFrames(scalar)
    # bin, and save stack
    hyperstack = binStack(hyperstack, binList)
    save_hyperstack(directory.binFile, hyperstack, "TZCYX")


def makeMasks(directory, rowDict):
    """
    following code block in for loop will iterate over unpacked samples and
    generate masks for each stimulus-paired sample using aitools learning
    well, it will in theory. At some point this will work... ¯\_(ツ)_/¯
    """
    # mask = load_hyperstack(directory.projFile)[0, 0, -1]
    # maskFile = Filepath(directory.maskDir, rowDict[MAIN["lay"]], ext="tif")
    # if not maskFile:
    #     save_hyperstack(maskFile, mask, shape="YX")
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
    hyperstack = load_hyperstack(directory.imgFile)
    # instantiate stimulus object from stimulus file, count frames
    stimulus = Stimulus(directory.frameFile)
    dfrt = None
    dfmt = None
    for maskFile in directory.maskFiles:
        mask = load_hyperstack(maskFile)[0, 0, 0]
        response = Response(
            stimulus=stimulus, mask=mask,
            region=Filepath.stabilize(maskFile[-1]),
            reporter_names=directory.channels)
        save_hyperstack(maskFile, response.labeledMask, shape="YX")
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
        dfn.insert(2, MAIN["cel"], rowDict[MAIN["cel"]])
        dfn.insert(3, MAIN["zpl"], rowDict[MAIN["zpl"]])
        # print(rowDict[MAIN["lay"]], stimulus.epochFrames)
        # if dfc.shape[1] < 80 or rowDict[MAIN["lay"]] is None:
        #     continue
        # print(dfc.shape)
        # fixed = np.linspace(0, 78, num=stimulus.epochFrames, endpoint=True).astype(
        #     int).astype(str).tolist()
        # if len(stimulus.xLabels) != 79:
        #     dfi = dfi.rename(columns=dict(zip(stimulus.numCols, fixed)))
        dfnt = (
            dfn if dfnt is None else pd.concat(
                (dfnt, dfn), axis=0, ignore_index=True))

    if dfnt is not None:
        # dfnt[(dfnt[MAIN["cel"]] == "L2 Tm3") & (dfnt[RESP["reg"]] == "M1")][RESP["reg"]] = "M5"
        # dfnt[(dfnt[MAIN["cel"]] == "L2 Tm3") & (dfnt[RESP["reg"]] == "M5")][MAIN["cel"]] = "Tm3"
        # dfnt[(dfnt[MAIN["cel"]] == "L2 Tm3") & (dfnt[RESP["reg"]] == "M2")][MAIN["cel"]] = "L2"
        # dfnt.loc[(dfnt[RESP["reg"]] == "M9-M10"), RESP["reg"]] = "M9 M10"
        # dfnt.loc[(dfnt[RESP["reg"]] == "M9 10um Thapsigargin"), RESP["reg"]] = "M9 + Tg"
        # dfnt.loc[(dfnt[RESP["reg"]] == "Lobula 10um Thapsigargin"), RESP["reg"]] = "Lobula + Tg"
        # cols = [str(x) for x in range(79)]
        # cons = dfnt.columns.values.tolist()[:11]
        # dfnt = dfnt[cons + cols]
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
        dfi.insert(2, MAIN["cel"], rowDict[MAIN["cel"]])
        dfi.insert(3, MAIN["zpl"], rowDict[MAIN["zpl"]])
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
        dfi.insert(2, MAIN["cel"], rowDict[MAIN["cel"]])
        dfi.insert(3, MAIN["zpl"], rowDict[MAIN["zpl"]])
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

    _save_pil_array(str(directory.avgFig), figures)


def plotAggregateResponses(directory):
    if not directory.totMeasFile:
        return

    figures = []
    stimulus = Stimulus(directory.frameFile)
    dft = pd.read_csv(directory.totMeasFile)
    dfc = dft[[
        MAIN["cel"], MAIN["dat"], MAIN["fly"], MAIN["zpl"],
        RESP["roi"]]].copy()
    dft = dft[[MAIN["cel"], RESP["reg"], RESP["chn"]] + stimulus.numCols]
    dft = dft.rename(
        columns=dict(zip(stimulus.numCols, stimulus.xLabels.astype(str))))
    chn = sorted(dft[RESP["chn"]].unique().tolist())
    col = dict(RGECO="#e93323", ER210="#6ace3d", ER150="#4eadea")
    for x in dft[MAIN["cel"]].unique().tolist():
        dfs = dft[dft[MAIN["cel"]] == x].copy().drop(columns=[MAIN["cel"]])
        dfn = dfc[dfc[MAIN["cel"]] == x].copy().drop(columns=[MAIN["cel"]])
        reg = sorted(dfs[RESP["reg"]].unique().tolist())
        chn = sorted(dfs[RESP["chn"]].unique().tolist())
        fig, axes = getFigAndAxes(rows=len(reg), cols=len(chn), sharex=True)
        for r, region in enumerate(reg):
            for c, channel in enumerate(chn):
                dfl = dfs[
                    (dfs[RESP["reg"]] == region) &
                    (dfs[RESP["chn"]] == channel)].copy()
                if dfl.empty:
                    clearAx(axes[r, c])
                else:
                    dfl = dfl.drop(columns=[RESP["reg"], RESP["chn"]]).melt(
                        var_name="time (s)", value_name="ΔF/F")
                    dfl["time (s)"] = dfl["time (s)"].astype(float)
                    linePlot(
                        ax=axes[r, c], data=dfl, xCol="time (s)", yCol="ΔF/F",
                        color=col[channel])
                    adjustNBins(axes[r, c])
                    redrawAxis(axes[r, c], xaxis=True)
                    shadeVerticalBox(
                        axes[r, c], start=stimulus.onTime,
                        stop=stimulus.offTime)

                if r == 0:
                    addHorizontalAxTitle(axes[r, c], channel)
                if c > 0:
                    addAxisLabels(axes[r, c], ylabel="")
                if r == len(reg) - 1:
                    specifyAxisTicks(
                        axes[-1, c], xticks=stimulus.customTicks(
                            bins=5, end=True))

            addVerticalAxTitle(axes[r, -1], region)

        figures += [figToImage("", fig)]

    _save_pil_array(str(directory.totAvgFig), figures)


def plotAggregateStatistics(directory):
    if not directory.totIntFile or not directory.totCorFile:
        return

    figures = []
    dfi = pd.read_csv(directory.totIntFile)
    dfp = pd.read_csv(directory.totCorFile, usecols=[
        MAIN["cel"], RESP["reg"], RESP["prv"], RESP["ppv"]])
    dfc = dfi[[
        MAIN["cel"], MAIN["dat"], MAIN["fly"], MAIN["zpl"],
        RESP["roi"], RESP["reg"]]].copy()
    dfi = dfi[[MAIN["cel"], RESP["reg"], RESP["chn"], RESP["int"]]]
    chn = sorted(dfi[RESP["chn"]].unique().tolist())
    col = dict(RGECO="#e93323", ER210="#6ace3d", ER150="#4eadea")
    pal = {RESP["prv"]: "#3c0751", RESP["ppv"]: "#468e8b"}
    for x in dfi[MAIN["cel"]].unique().tolist():
        dfis = dfi[dfi[MAIN["cel"]] == x].copy().drop(columns=[MAIN["cel"]])
        dfps = dfp[dfp[MAIN["cel"]] == x].copy().drop(columns=[MAIN["cel"]])
        dfcs = dfc[dfc[MAIN["cel"]] == x].copy().drop(columns=[MAIN["cel"]])
        reg = sorted(dfis[RESP["reg"]].unique().tolist())
        fig, axes = getFigAndAxes(rows=len(reg), cols=3)
        for r, region in enumerate(reg):
            dfisr = dfis[dfis[RESP["reg"]] == region].copy().drop(
                columns=[RESP["reg"]])
            dfcsr = dfcs[dfcs[RESP["reg"]] == region].copy().drop(
                columns=[RESP["reg"]])
            if dfisr.empty:
                [clearAx(axes[r, x]) for x in range(3)]
            else:
                barPlot(
                    ax=axes[r, 0], data=dfisr, cCol=RESP["chn"],
                    yCol=RESP["int"], hueDict=col, raw=True, order=sorted(col))
                dfpsr = dfps[dfps[RESP["reg"]] == region].copy().drop(
                    columns=[RESP["reg"]]).rename(
                    columns={RESP["prv"]: "R", RESP["ppv"]: "p-value"}).melt(
                    var_name="statistic", value_name="value")
                adjustNBins(axes[r, 0])
                barPlot(
                    ax=axes[r, 1], data=dfpsr, cCol="statistic", yCol="value",
                    raw=False, order=["R", "p-value"])
                annotatePatches(ax=axes[r, 1])
                resetLimits(axes[r, 1], ys=[-1, 1])
                fN, zN, rN = countNs(dfcsr)
                data = pd.DataFrame(dict(flies=[fN], planes=[zN], ROIs=[rN]))
                data = np.log(data)
                data = data.melt(var_name="quantity", value_name="log N")
                barPlot(
                    ax=axes[r, 2], data=data, cCol="quantity", yCol="log N",
                    raw=False, ci=None, order=["flies", "planes", "log N"])
                for a in range(3):
                    redrawAxis(axes[r, a], xaxis=True)

            addVerticalAxTitle(axes[r, -1], region)

        addHorizontalAxTitle(axes[0, 0], "integrated responses")
        addHorizontalAxTitle(axes[0, 1], "pearson's correlation")
        addHorizontalAxTitle(axes[0, 2], "N")
        figures += [figToImage("", fig)]

    _save_pil_array(str(directory.totStatFig), figures)


def countNs(df):
    df = df[[MAIN["dat"], MAIN["fly"], MAIN["zpl"], RESP["roi"]]]
    df = df.drop_duplicates(keep="first")
    rN = df.shape[0]
    df = df.drop(columns=[RESP["roi"]]).drop_duplicates(keep="first")
    zN = df.shape[0]
    df = df.drop(columns=[MAIN["zpl"]]).drop_duplicates(keep="first")
    fN = df.shape[0]
    return fN, zN, rN
