#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon March 7 11:41:02 2022

@author: ike
"""


from ..utils.filepath import Filepath


MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec"]


class Directory(object):
    """
    Convenience functions for creating or locating filepaths, directory
    structures, etc. Can be modified to accomodate any nonambiguous directory
    structure that contains all images acquired from the same coordinates (ie.
    multiple stimuli presented to the same z-plane for two photon imaging)
    """
    def __init__(
            self, dataDir, cell, date, fly, zplane, stimName=None,
            stimTime=None, LIFName=None, ch0=None, ch1=None):
        stimName = str(stimName)
        LIFName = str(LIFName)
        stimTime = ("None" if stimTime is None else int(stimTime))
        self.sampDir = Filepath(
            dataDir, cell, "{:d}-Fly{:d}-{}-Z{:d}".format(
                int(date), int(fly), cell, int(zplane)))
        self.imgDir = Filepath(self.sampDir, stimName)
        self.maskDir = Filepath(self.sampDir, "masks")
        self.measDir = Filepath(self.sampDir, "measurements")
        self.figDir = Filepath(self.sampDir, "figures")
        self.totMeasDir = Filepath(dataDir, "measurements")
        self.totFigDir = Filepath(dataDir, "figures")

        self.lifFile = Filepath(
            dataDir, "**", "*{}*".format(LIFName), ext="lif").search(
                recursive=True, first=True)
        self.channels = [
            c for c in (ch0, ch1) if type(c) == str and str(c) != "nan"]
        self.imgFile = Filepath(
            self.imgDir, " ".join([stimName] + self.channels), ext="tif")
        self.binFile = Filepath(
            self.imgFile.astype(None) + "-binned", ext="tif")
        self.stimFile = Filepath(
            dataDir, "**", "*{}*{}*{}".format(
                stimName, self.stimDate(date), stimTime),
            ext="csv").search(recursive=True, first=True)
        self.frameFile = (
            Filepath(self.stimFile.astype(None) + "-frames", ext="csv")
            if self.stimFile is not None else None)

        self.projFile = Filepath(self.sampDir, "average projection", ext="tif")
        self.rawFile = Filepath(
            self.measDir, "{}-raw".format(stimName), ext="csv")
        self.measFile = Filepath(
            self.measDir, "{}-measurements".format(stimName), ext="csv")
        self.indFile = Filepath(
            self.measDir, "{}-individual".format(stimName), ext="csv")
        self.avgFile = Filepath(
            self.measDir, "{}-average".format(stimName), ext="csv")
        self.rfcFile = Filepath(self.measDir, "RF centers", ext="csv")
        self.maskFiles = Filepath(self.maskDir, "*", ext="tif").search()

        self.totMeasFile = Filepath(
            self.totMeasDir, "{}-{}-measurements".format(
                cell, stimName), ext="csv")
        self.totIntFile = Filepath(
            self.totMeasDir, "{}-{}-integrated".format(
                cell, stimName), ext="csv")
        self.totCorFile = Filepath(
            self.totMeasDir, "{}-{}-correlations".format(
                cell, stimName), ext="csv")

        self.avgFig = Filepath(
            self.figDir, "{}-average responses".format(stimName), ext="tif")

        self.totAvgFig = Filepath(
            self.totFigDir, "{}-{}-average responses".format(
                cell, stimName), ext="tif")
        self.totIntFig = Filepath(
            self.totFigDir, "{}-{}-integrated responses".format(
                cell, stimName), ext="tif")
        self.totCorFig = Filepath(
            self.totFigDir, "{}-{}-correlated responses".format(
                cell, stimName), ext="tif")


    @staticmethod
    def stimDate(date):
        date = str(int(date))
        return "_".join((date[:4], MONTHS[int(date[4:6]) - 1], date[6:]))
