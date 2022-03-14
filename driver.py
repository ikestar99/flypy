#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 03:13:56 2021

@author: ike
"""


import flypy.examples.calciumimaging as fec
from flypy.utils.pathutils import *
from flypy.utils.hyperstack import *
from flypy.examples.katherinealignment import alignImagesInLIFFile


# fec.pipeline()


# for file in glob("/Users/ike/Desktop/test copy/*/*/average projection.tif"):
    # layer = file[file.rindex("Z") + 3:file.rindex("/")]
    # if layer != "WHOLE":
    #     average = loadTZCYXTiff(file)[:,:,-1][:,:,np.newaxis]
    #     average = average * int(255 / np.max(average))
    #     save = getPath(getParent(file), "masks", "background", ext="tif")
    #     saveTZCYXTiff(save, average, shape="TZCYX")
    #     save = getPath(getParent(file), "masks", layer, ext="tif")
    #     saveTZCYXTiff(save, average, shape="TZCYX")


# alignImagesInLIFFile("/Users/ike/Desktop/20220305_fly2.lif", [5, 6], 1)