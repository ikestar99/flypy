#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import numpy as np

from readlif.reader import LifFile


def loadLifFile(file):
    """
    Load entire lif file as an object from which to extract imaging samples

    @param file: path to .lif file to be loaded
    @type file: string

    @return: LifFile iterable that contains all imaged samples in lif file
    @rtype: readlif.reader.LifFile
    """
    lif = LifFile(file)
    return lif


def getLifImage(lif, idx, dtype=np.uint8):
    """
    Extract an imaged sample as a hyperstack from a pre-loaded .lif file

    @param lif: LifFile iterable that contains all imaged samples in lif file
    @type lif: readlif.reader.LifFile
    @param idx: index of desired image within lif file (0-indexed)
    @type idx: int
    @param dtype: data type of image array to be saved
    @type dtype: np.dtype, passed to np.ndarray.astype constructor

    @return: lif imaging sample converted to 5D hyperstack array
    @rtype: numpy.ndarray
    """
    image = lif.get_image(img_n=idx)
    stack = [[[np.array(image.get_frame(t=t, z=z, c=c))
               for c in range(image.channels)]
              for z in range(image.dims.z)]
             for t in range(image.dims.t)]
    stack = np.array(stack, dtype=dtype)
    return stack
