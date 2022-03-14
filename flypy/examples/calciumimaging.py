#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 7 11:41:02 2022

@author: ike
"""


import numpy as np
import pandas as pd
import os.path as op

from ..main import tqdm
from ..utils.pathutils import *
from ..utils.pipeutils import boundInt, smooth
from ..utils.csvreader import CSVReader
from ..utils.hyperstack import (
    saveTZCYXTiff, loadTZCYXTiff, savePillowArray)
from ..pipeline.roi import (
    binStack, labelMask, extractMaskedValuePerFrame, makeFigure)
from ..pipeline.liffile import loadLifFile, getLifImage
from ..pipeline.stimulus import Stimulus, FRM, RTM, ENM
from ..pipeline.alignment import alignStack


# directory that contains ALL files to be analyzed (.lif, .csv, .tif)
DIRECTORY = "/Users/ike/Desktop/test copy"
# path to imaging.csv file with desired settings
CSVPATH = "/Users/ike/Desktop/test/New Pipeline Imaging Settings.csv"
# column keys in imaging.csv file
DAT = "Date"
FLY = "Fly"
CEL = "Cell"
LAY = "Layers"
ZPL = "Z Plane"
LIF = ".lif Name"
LNM = ".lif Num"
STN = "Stim Name"
STT = "Stim Timestamp"
CH0 = "Channel 0"
CH1 = "Channel 1"
BLO = "Begin Baseline"
BL1 = "End Baseline"
REF = "Reference Channel"
# column keys in response.csv files
ROW = "sample"
ROI = "ROI"
CHN = "channel"
AVG = "mean_PI"
SIZ = "sizes"
RNM = "response_number"
NRS = "response N"
# number of channels per image
CHANNELS = 2
# reciprocal multiple of frequency at which to bin images
BINSCALAR = 2
# string in name of background mask for background correction
BKG = "background"


"""
Convenience functions for creating or locating filepaths, directory structures,
etc. Can be modified to accomodate any nonambiguous directory structure that
contains all images acquired from the same coordinates (ie. multiple stimuli
presented to the same z-plane for two photon imaging)
"""


def getRowDirectory(rowDict):
    """
    Convert row dict from CSVReader object into directory name of
    corresponding sample

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to directory for given row sample. Directories of format:
        DIRECTORY/cell/date-Fly#-Z#-layer
    @rtype: string
    """
    sampleDirectory = getPath(
        DIRECTORY, cleanName(rowDict[CEL]), cleanName("-".join((
            str(rowDict[DAT]), "".join(
                ("Fly", str(rowDict[FLY]))), str(rowDict[CEL]),
            "".join(("Z", str(rowDict[ZPL]))), str(rowDict[LAY])))))
    return sampleDirectory


def getImageDirectory(rowDict):
    """
    Convert row dict from CSVReader object into directory name of
    corresponding image stack

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to image directory for given row sample
    @rtype: string
    """
    imageDirectory = getPath(getRowDirectory(rowDict), cleanName(rowDict[STN]))
    return imageDirectory


def getMaskDirectory(rowDict):
    """
    Convert row dict from CSVReader object into directory name of mask folder

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to image directory for given row sample
    @rtype: string
    """
    maskDirectory = getPath(getRowDirectory(rowDict), "masks")
    return maskDirectory


def getMeasurementDirectory(rowDict):
    """
    Convert row dict from CSVReader object into directory name of measurements
    folder

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to measurements directory for given row sample
    @rtype: string
    """
    measurementDirectory = getPath(getRowDirectory(rowDict), "measurements")
    return measurementDirectory


def getFigureDirectory(rowDict):
    """
    Convert row dict from CSVReader object into directory name of figures
    folder

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to figures directory for given row sample
    @rtype: string
    """
    figureDirectory = getPath(getRowDirectory(rowDict), "figures")
    return figureDirectory


def getLifFile(lifName):
    # ensure lifName has .lif file extension
    lifFile = firstGlob(
        DIRECTORY, "**", "".join(("*", changeExt(lifName), "*")), ext="lif")
    return lifFile


def getImagePath(rowDict):
    """
    Convert row dict from CSVReader object into filepath of image stack

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to image for given row sample
    @rtype: string
    """
    imageDirectory = getPath(
        getImageDirectory(rowDict), cleanName(rowDict[STN]), ext="tif")
    return imageDirectory


def getBinnedPath(rowDict):
    """
    Convert row dict from CSVReader object into filepath of binned image stack

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to binned image for given row sample
    @rtype: string
    """
    binnedDirectory = getPath(
        getImageDirectory(rowDict),
        "-".join((cleanName(rowDict[STN]), "binned")), ext="tif")
    return binnedDirectory


def getOldStimFile(rowDict):
    """
    Find path to stim file corresponding to current row dict and derive new
    file paath

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: old path to stim file, new path to stim file
    @rtype: string, string
    """
    if type(rowDict[STN]) != str or "nan" in (
            str(rowDict[STT]), str(rowDict[STN])):
        return None

    oldStim = firstGlob(DIRECTORY, "**", "".join(
        ("*", str(rowDict[STN]), "*", str(int(rowDict[STT])))), ext="csv")
    return oldStim


def getNewStimFile(rowDict):
    """
    Find path to stim file corresponding to current row dict and derive new
    file paath

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: old path to stim file, new path to stim file
    @rtype: string, string
    """
    oldStim = getOldStimFile(rowDict)
    newStim = (None if oldStim is None else getPath(
        getImageDirectory(rowDict), getName(oldStim)))
    return newStim


def getFrameFile(rowDict):
    """
    Find path to frame-counted stim file corresponding to current row dict

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to stim frame file
    @rtype: string
    """
    newStim = getNewStimFile(rowDict)
    frameStim = (None if newStim is None else changeExt(
        "-".join((changeExt(newStim), FRM)), "csv"))
    return frameStim


def getRawFile(rowDict):
    """
    Find path to raw responses.csv file corresponding to current row dict

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to raw responses.csv
    @rtype: string
    """
    rawFile = getPath(
        getMeasurementDirectory(rowDict),
        "-".join((cleanName(rowDict[STN]), "raw")), ext="csv")
    return rawFile


def getMeasuementsFile(rowDict):
    """
    Find path to measurements.csv file corresponding to current row dict

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to measurements.csv
    @rtype: string
    """
    measurementsFile = getPath(
        getMeasurementDirectory(rowDict),
        "-".join((cleanName(rowDict[STN]), "measurements")), ext="csv")
    return measurementsFile


def getIndividualFile(rowDict):
    """
    Find path to individual responses.csv file corresponding to current row
    dict

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to individual responses.csv
    @rtype: string
    """
    individualFile = getPath(
        getMeasurementDirectory(rowDict),
        "-".join((cleanName(rowDict[STN]), "individual")), ext="csv")
    return individualFile


def getAverageFile(rowDict):
    """
    Find path to average responses.csv file corresponding to current row
    dict

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to average responses.csv
    @rtype: string
    """
    averageFile = getPath(
        getMeasurementDirectory(rowDict),
        "-".join((cleanName(rowDict[STN]), "average")), ext="csv")
    return averageFile


def getAveragePath(rowDict):
    """
    Convert row dict from CSVReader object into filepath of average projection

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to average projection for given row sample
    @rtype: string
    """
    averagePath = getPath(
        getRowDirectory(rowDict), "average projection", ext="tif")
    return averagePath


def getAverageFigure(rowDict):
    """
    Convert row dict from CSVReader object into filepath of average figure

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: path to average figure for given row sample
    @rtype: string
    """
    averageFigure = getPath(
        getFigureDirectory(rowDict), "average responses", ext="tif")
    return averageFigure


def getMaskPaths(rowDict):
    """
    Convert row dict from CSVReader object into list of mask filepaths

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: list of paths to masks for given row sample
    @rtype: list
    """
    maskPaths = glob(getMaskDirectory(rowDict), "*", ext="tif")
    return maskPaths


def getChannels(rowDict):
    """
    Convert row dict from CSVReader object into ordered list of indicators
    imaged in given row

    @param rowDict: row from imaging CSVReader
    @type rowDict: dict

    @return: list of ordered indicator channels for given row sample
    @rtype: string, string
    """
    ch0 = rowDict[CH0]
    ch1 = rowDict[CH1]
    return ch0, ch1


def pipeline():
    # load imaging settings CSV file
    data = CSVReader(CSVPATH)
    """
    following code block in for loop will unpack lif files and form appropriate
    directory structure for all samples to be analyzed, skipping any samples
    for which the procedure is already complete
    """
    for lifName in tqdm(data.getColumnSet(LIF)):
        lifFile = getLifFile(lifName)
        if lifFile is None:
            print("{} not found".format(lifFile))
            continue

        # load .lif file as object from which to extract imaged samples
        lifFile = loadLifFile(lifFile)
        # iterate over all samples in this lif file
        for rowDict in data[{LIF: lifName}]:
            # file paths for stim file, frame stim file, raw data, directories
            rowDirectory = getRowDirectory(rowDict)
            imageDirectory = getPath(rowDirectory, cleanName(rowDict[STN]))
            subDirs = [
                rowDirectory, imageDirectory, getMeasurementDirectory(rowDict),
                getFigureDirectory(rowDict), getMaskDirectory(rowDict)]
            oldStim = getOldStimFile(rowDict)
            newStim = getNewStimFile(rowDict)
            hyperstackPath = getImagePath(rowDict)
            # create directory structure for current sample if not present
            for directory in subDirs:
                makeDir(directory)

            # extract image from lifFile object and save
            if not op.isfile(hyperstackPath):
                hyperstack = getLifImage(lifFile, idx=rowDict[LNM] - 1)
                saveTZCYXTiff(hyperstackPath, hyperstack, shape="TZCYX")
            # move associated stim file to imageDirectory
            if type(oldStim) == str and oldStim != newStim:
                movePath(oldStim, newStim)
            elif oldStim is None:
                # print("{} has no stim file".format(imageDirectory))
                pass

    """
    following code block in for loop will iterate over unpacked
    samples and align all images acquired from the same z-plane
    """
    for sample in tqdm(data(DAT, FLY, CEL, LAY, ZPL, CH0, CH1)):
        # find all rows from same date/fly/cell/layer/Z/indicators
        rowDicts = data[sample]
        # file paths for raw data, average projection
        hyperstackPaths = sorted([getImagePath(r) for r in rowDicts])
        averagePath = getAveragePath(rowDicts[0])
        # if images already aligned, skip current sample
        if op.isfile(averagePath):
            continue

        # load each image as hyperstack
        hyperstacks = [loadTZCYXTiff(p) for p in hyperstackPaths]
        # record number of frames in each hyperstack
        frames = [h.shape[0] for h in hyperstacks]
        # concatenate all images in sample for cross-stimulus alignment
        hyperstack = (
            np.concatenate(hyperstacks, axis=0) if len(hyperstacks) > 1
            else hyperstacks[0])
        # align hyperstack three times
        for _ in range(3):
            hyperstack = alignStack(
                hyperstack, channel=int(rowDicts[0][REF]), mode="rigid")

        # split alighned hyperstacks into original rows and save each one
        frame = 0
        for x, hyperstackPath in enumerate(hyperstackPaths):
            aligned = hyperstack[frame:frame + frames[x]]
            saveTZCYXTiff(hyperstackPath, aligned, "TZCYX")
            frame += frames[x]

        # create and save average projection of aligned hyperstack
        hyperstack = np.mean(hyperstack, axis=0, keepdims=True)
        saveTZCYXTiff(averagePath, hyperstack, "TZCYX")

    """
    following code block in for loop will iterate over unpacked
    samples and bin all aligned images
    """
    for rowDict in tqdm(data):
        # file paths for stim file, frame stim file, raw data, and binned data
        newStim = getOldStimFile(rowDict)
        frameStim = getFrameFile(rowDict)
        hyperstackPath = getImagePath(rowDict)
        binnedPath = getBinnedPath(rowDict)
        # if images already binned, skip current sample
        if newStim is None or op.isfile(binnedPath):
            continue

        # instantiate stimulus object from stimulus file, count frames
        stimulus = (
            Stimulus(frameStim) if op.isfile(frameStim) else Stimulus(newStim))
        # save frame stim if not already saved
        (None if op.isfile(frameStim) else stimulus.dfs.to_csv(
            frameStim, encoding="utf-8", index=False))
        # extract list if images/bin from stimulus object
        binList, _ = stimulus.binFrames(BINSCALAR)
        # load, bin, and save stack associated with current row
        hyperstack = loadTZCYXTiff(hyperstackPath)
        hyperstack = binStack(hyperstack, binList)
        saveTZCYXTiff(binnedPath, hyperstack, "TZCYX")

    """
    TODO: check all code below this marker. All upstream code verified and
        functional as of 20220312 20:50:00, verified by Ike Ogbonna
    """
    import sys; sys.exit()
    """
    following code block in for loop will iterate over unpacked samples and
    generate masks for each stimulus-paired sample using machine learning
    well, it will in theory. At some point this will work... ¯\_(ツ)_/¯
    """
    for rowDict in tqdm(data):
        pass

    """
    following code block in for loop will iterate over unpacked samples and
    measure raw, individual, and average responses per ROI in each specified
    mask
    """
    for rowDict in tqdm(data):
        # file paths for frame stim file, raw data, masks, response csv files
        frameStim = getFrameFile(rowDict)
        hyperstackPath = getImagePath(rowDict)
        maskPaths = getMaskPaths(rowDict)
        rawFile = getRawFile(rowDict)
        measurementsFile = getMeasuementsFile(rowDict)
        individualFile = getIndividualFile(rowDict)
        averageFile = getAverageFile(rowDict)
        # if responses already measured or frames not counted, skip
        if op.isfile(averageFile) or not op.isfile(str(frameStim)):
            continue

        # instantiate stimulus object from stimulus file
        stimulus = Stimulus(frameStim)
        # empty holders for response data dataframe and background
        dfr = None  # raw response holder
        dfi = None  # individual response holder
        background = None
        # dataframe column headers for raw and per-epoch responses
        rawColumns = [str(x + 1) for x in range(stimulus.totalFrames)]
        subColumns = [str(float(t)).rstrip("0").rstrip(".") for t in np.arange(
            0, stimulus.epochTime, stimulus.frameTime)[:int(
                stimulus.epochFrames)]]
        # load raw data
        hyperstack = loadTZCYXTiff(hyperstackPath)
        # identify background mask and extract background values per frame
        for x, mask in enumerate(maskPaths):
            if BKG in mask:
                mask = loadTZCYXTiff(mask)
                background = extractMaskedValuePerFrame(hyperstack, mask)
                # background array shape "TZC" -squeeze-> "TC" -moveaxis-> "CT"
                background = np.moveaxis(np.squeeze(background), -1, 0)
                del maskPaths[x]
                break

        # skip sample if no background mask found
        if background is None:
            print("{} has no associated background".format(hyperstackPath))
            continue

        # extract background-corrected responses from every ROI in every mask
        """
        code block equivalent to former "measure raw responses" portion of
        automated endopy pipeline
        """
        for path in maskPaths:
            # load mask, label and count ROIs, save labeled image
            mask, regions = labelMask(loadTZCYXTiff(path))
            saveTZCYXTiff(path, mask, "TZCYX")
            # first array identifies ROI, second specifies ROI size
            rois, sizes = np.unique(mask[np.nonzero(mask)], return_counts=True)
            # response array shape "TZC" -squeeze-> "TC" -moveaxis-> "CT"
            responses = np.concatenate(
                [np.moveaxis(np.squeeze(
                    extractMaskedValuePerFrame(hyperstack, mask == r)), -1, 0)
                 for r in range(regions)], axis=0)
            # duplicate ROI number, size, and identify indicator per channel
            rois = np.repeat(rois, CHANNELS)
            sizes = np.repeat(sizes, CHANNELS)
            channels = [getChannels(rowDict)] * regions
            # convert responses array into pandas dataframe
            dfs = pd.DataFrame(data=responses, columns=rawColumns)
            # add average raw response column to dataframe
            dfs.insert(0, AVG, np.mean(responses, axis=-1))
            # add sample name column to dataframe
            dfs.insert(0, ROW, getName(getRowDirectory(rowDict)))
            # add cell-layer column to dataframe
            dfs.insert(1, LAY, "-".join((rowDict[CEL], rowDict[LAY])))
            # add ROI number column to dataframe
            dfs.insert(2, ROI, rois)
            # add channels column to dataframe
            dfs.insert(3, CHN, channels)
            # add size colum to dataframe
            dfs.insert(4, SIZ, sizes)
            # add mask-specific dataframe to total dataframe
            dfr = (dfs if dfr is None else pd.concat(
                (dfr, dfs), axis=0, ignore_index=True))

        # save raw and measurement csv files
        dfr.to_csv(rawFile, encoding="utf-8", index=False)
        dfr[[ROW, LAY, ROI, SIZ]].copy().drop_duplicates().to_csv(
            measurementsFile, encoding="utf-8", index=False)

        """
        code block equivalent to former "measure individual responses" portion
        of automated endopy pipeline
        """
        # subtract background from raw responses
        dfr[rawColumns] = dfr[rawColumns].to_numpy() - background
        # extract median relative stimulus time per frame
        rel_time = stimulus[RTM]
        # epoch first index, first frame of each epoch
        efi = rel_time[:-1] > (rel_time[1:] + (stimulus.epochTime / 2))
        efi = list(set([0] + (np.nonzero(efi)[0] + 1).tolist()))
        eln = stimulus[ENM]
        # efi[0] = efi[1] - row("eFrames")
        # except IndexError:
        #     continue

        # Find, resample, normalize, save responses to single epochs
        for e, edx in enumerate(efi):
            dft = dfr[[ROW, LAY, ROI, CHN, SIZ]].copy()
            # add response number column to dataframe
            dft.insert(3, RNM, e + 1)
            # add epoch number column to dataframe
            dft.insert(3, ENM, eln[e])
            # extract background-corrected responses of current epoch
            currentColumns = [
                str(boundInt((edx + x), 1, stimulus.totalFrames))
                for x in range(stimulus.epochFrames)]
            epochResponses = dfr[currentColumns].to_numpy()
            # normalize responses to flexible baseline
            bln = np.median(
                epochResponses[:, rowDict[BLO]:rowDict[BL1]], axis=-1)[
                :, np.newaxis]
            epochResponses = (epochResponses - bln) / bln
            # add epoch-specific dataframe to total dataframe
            dft = pd.concat((dft, pd.DataFrame(
                epochResponses, columns=subColumns)), axis=1)
            dfi = (dft if dfi is None else pd.concat(
                (dfi, dft), axis=0, ignore_index=True))

        dfi = dfi.sort_values(
            [ROW, LAY, ROI, ENM, RNM, CHN], ascending=True, ignore_index=True)
        dfi.to_csv(individualFile, encoding="utf-8", index=False)

        """
        code block equivalent to former "measure average responses" portion
        of automated endopy pipeline
        """
        # average individual responses across epochs of same type
        dfa = dfi.drop(columns=[ENM], axis=1).groupby(
            [ROW, LAY, ROI, ENM, CHN]).mean().reset_index()
        # add response number column to dataframe
        dfa.insert(4, ENM, dfi[RNM].max())
        dfa.to_csv(averageFile, encoding="utf-8", index=False)

        """
        code block equivalent to former "plot average responses" portion of
        automated endopy pipeline
        """
        averageFigure = getAverageFigure(rowDict)
        figures = []
        dfa[[ROW]] = (
            dfa[[ROW]].astype(str) + " " + dfa[[LAY]] + " ROI " +
            dfa[[ROI]].astype(str) + " Responses " + dfa[[NRS]].astype(str))
        for name in dfa[[ROW]].unique().tolist():
            dfr = dfa[dfa[[ROW]] == name].copy()
            avg = dict
            sub = dfr[[CHN]].unique().tolist()
            for c in sub:
                for e in dfr[[ENM]].unique().tolist():
                    responses = dfr[(dfr[[CHN]] == c) & (dfr[[ENM]] == e)][
                        subColumns].to_numpy()[0]
                    avg = {" ".join((str(c), str(e))): smooth(responses)}

            figures += [makeFigure(
                X=np.array(subColumns, dtype=float), Ys=avg,
                titles=(name, "Time (s)", "Average ΔF/F"),
                light=[0.25 * stimulus.epochTime, 0.75 * stimulus.epochTime],
                subs=sub)]

        savePillowArray(averageFigure, figures)
