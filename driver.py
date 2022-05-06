#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 03:13:56 2021

@author: ike
"""


import flypy.examples.calciumimaging as fec
import flypy.utils.csvreader as csv
from flypy.utils.csvcolumns import MAIN


# directory that contains ALL files to be analyzed (.lif, .csv, .tif)
DIRECTORY = "/Users/ike/Desktop/test copy"
# path to imaging.csv file with desired settings
CSVPATH = "/Users/ike/Desktop/test copy/New Pipeline Imaging Settings.csv"


def __main__():
    imagingCSVReader = csv.CSVReader.fromFile(CSVPATH)
    imagingCSVReader.dropna(MAIN["dat"])
    # fec.calciumImagingPreparation(DIRECTORY, imagingCSVReader)
    # fec.calciumImagingPipeline(DIRECTORY, imagingCSVReader)
    # fec.responseAggregation(DIRECTORY, imagingCSVReader)
    # fec.integrateAndCorrelate(DIRECTORY, imagingCSVReader)
    fec.plotAggregation(DIRECTORY, imagingCSVReader)


if __name__ == "__main__":
    __main__()
