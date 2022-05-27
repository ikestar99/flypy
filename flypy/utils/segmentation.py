#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import numpy as np
import scipy.ndimage as scn


def segment_ROIs(mask):
    """
    Label all ROIs in a given mask with a unique integer value

    @param mask: integer mask array on which to perform labeling
    @type mask: numpy.ndarray

    @return: mask array with all unique non-zero ROIs labeled with a unique
        integer

        second returned value is number of ROIs labeled
    @rtype: numpy.ndarray, int
    """
    labeledMask, regions = scn.label(mask)
    return labeledMask, regions


def generate_ROI_mask(mask, region):
    return (mask == region).astype(int)


def measureROIFs(raw, mask):
    """
    Compute mean pixel intensity in masked region of each image in a stack
    or hyperstack

    @param raw: input array of images on which to extract mean value
        of masked region, hyperstack or otherwise. Should correspond to a
        single ROI
    @type raw: numpy.ndarray
    @param mask: 2D mask of region within image from which to compute
        mean
    @type mask: numpy.ndarray

    @return: array of same shape as input array with the omission of the
        last two axes corresponding to YX dimensions, which have been
        collapsed into a single integer
    @rtype: numpy.ndarray
    """
    mask = (mask.astype(int) > 0).astype(float)
    ROISize = np.sum(mask)
    maskedArray = raw * mask
    maskedArray = np.sum(maskedArray, axis=(-2, -1)).astype(float)
    maskedArray = (maskedArray / ROISize).astype(float)
    return maskedArray


def subtractBackground(raw, background):
    """
    Compute mean pixel intensity in background region of each image in a
    stack or hyperstack and subtract each image-specific value from all
    pixels in the corresponding image

    @param raw: input array of images on which to perform background
        correction, hyperstack or otherwise
    @type raw: numpy.ndarray
    @param background: 2D mask of region within image from which to compute
        background
    @type background: numpy.ndarray

    @return: array of same shape as input array with the average background
        pixel intensity of each 2D image subtracted from all other pixels
        within the image
    @rtype: numpy.ndarray
    """
    background = (background > 0).astype(int)
    background = measureROIFs(raw, background)
    background = background[..., np.newaxis, np.newaxis]
    rawstack = raw - background
    return rawstack


def measure_multiple_ROIs(raw, mask):
    labeledMask, regions = segment_ROIs(mask)
    ROIs = [measureROIFs(raw, generate_ROI_mask(mask, r)) for r in regions]
    return ROIs, labeledMask, regions
