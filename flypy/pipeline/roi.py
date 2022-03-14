#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import io
from PIL import Image
import numpy as np
import seaborn as sns
import scipy.ndimage as scn
import matplotlib.pyplot as plt


def binStack(array, bins):
    """
    Bin an imaging time series according to a known list of frames per bin

    @param array: input array of images to be binned, hyperstack or otherwise.
        First axis of array must correspond to time dimension, as binning
        occurs along the first axis
    @type array: numpy.ndarray
    @param bins: list of imaging frames in each bin. Each index in bins is a
        sublist of all frames that should be averaged to yield the index-th
        frame in the binned image. ie the following list of lists:
        [[1, 5, 8, 9]
         [2, 3, 6, 10]
         [4, 7, 11]]
        indicates that there are 3 binned frames from an imaging array with
        11 unbinned frame. The first binned image in this example includes all
        frames in bins[0]: 1, 5, 8, and 9
    @type bins: list

    @return: binned image of same shape as input array except in axis 0
    @rtype: numpy.ndarray
    """
    # form a list of binned images with list comprehension
    bins = [set([min(x, array.shape[0] - 1) for x in b]) for b in bins]
    binned = [np.mean(np.array(
        [array[t] for t in bin]), axis=0) for bin in bins]
    binned = np.array(binned, dtype=array.dtype)
    return binned


def labelMask(mask):
    """
    Label all ROIs in a given mask with a unique integer value

    @param mask: integer mask array on which to perform labeling
    @type mask: numpy.ndarray

    @return: mask array with all unique non-zero ROIs labeled with a unique
        integer

        second returned value is number of ROIs labeled
    @rtype: numpy.ndarray, int
    """
    mask, regions = scn.label(mask)
    return mask, regions


def extractMaskedValuePerFrame(array, mask):
    """
    Compute mean pixel intensity in maskedd region of each image in a stack
    or hyperstack

    @param array: input array of images on which to extract mean value of
    masked region, hyperstack or otherwise. Should correspond to a single ROI
    @type array: numpy.ndarray
    @param mask: 2D mask of region within image from which to compute
        mean
    @type mask: numpy.ndarray

    @return: array of same shape as input array with the omission of the last
        two axes corresponding to YX dimensions, which have been collapsed
        into a single integer
    @rtype: numpy.ndarray
    """
    mask = mask.astype(int) > 0
    while mask.ndim < array.ndim:
        mask = mask[np.newaxis]

    maskedArray = array * mask
    maskedArray = np.sum(maskedArray, axis=(-2, -1))
    maskedArray = (maskedArray / np.sum(mask)).astype(array.dtype)
    return maskedArray


def subtractBackground(array, background):
    """
    Compute mean pixel intensity in background region of each image in a stack
    or hyperstack and subtract each image-specific value from all pixels in the
    corresponding image

    @param array: input array of images on which to perform background
        correction, hyperstack or otherwise
    @type array: numpy.ndarray
    @param background: 2D mask of region within image from which to compute
        background
    @type background: numpy.ndarray

    @return: array of same shape as input array with the average background
        pixel intensity of each 2D image subtracted from all other pixels
        within the image
    @rtype: numpy.ndarray
    """
    background = extractMaskedValuePerFrame(array, background)
    background = background[..., np.newaxis, np.newaxis]
    array = array - background
    return array


def makeFigure(X, Ys, titles, dYs=None, light=None, subs=None):
    """
    generate an elegant, formated response figure for a particular ROI

    @param X: shared x-coordinates of curve to be graphed
    @type X: numpy.ndarray
    @param Ys: dictionary of {indicator: y-coordinates of specific indicator}
        pairings for all indicators to be plotted. Y-coordinates should be
        stored in numpy arrays
    @type Ys: dict
    @param titles: [figure title, x-axis label, y-axis label]
    @type titles: iterable
    @param dYs: dictionary of {indicator: std or variance} pairings for all
        indicators to be plotted. Y-coordinates should be stored in numpy
        arrays
    @type dYs: dict
    @param light: [stimulus on time, stimulus off time]
    @type light: list
    @param subs:
    @type subs:

    @return: response figure made with matplotlib, stored in a buffer, and
    converted into an image for performant use and easy storage
    @rtype: PIL.Image
    """
    plt.rcParams["font.size"] = "8"
    sns.set_style("darkgrid")
    subs = (list(Ys.keys()) if subs in None else subs)
    fig, ax = plt.subplots(
        nrows=len(subs), figsize=(4, (3 * len(subs))), sharex=True, dpi=150)
    ax = ([ax] if len(subs) == 1 else ax)
    n_colors = (len(Ys) if len(Ys) == len(subs) else len(Ys) // len(subs))
    col = sns.color_palette("rocket", n_colors=len(Ys))  # "viridis"
    for r, subKey in enumerate(subs):
        for c, key in enumerate(Ys):
            if subKey in key:
                ax[r].plot(X, Ys[key], label=key, lw=1, color=col[r])
                if dYs is not None and key in dYs:
                    ax[r].fill_between(
                        X, Ys[key] - dYs[key], Ys[key] + dYs[key], color=col[r],
                        alpha=0.5)

        if ax[r].get_ylim()[0] <= 0 <= ax[r].get_ylim()[-1]:
            ax[r].hlines(y=0, xmin=X[0], xmax=X[-1], lw=1, color='black')
        if light is not None:
            plt.axvspan(light[0], light[1], color="blue", lw=1, alpha=0.1)

        ax[r].set_xlabel(titles[1])
        ax[r].set_ylabel(titles[2])
        ax[r].legend(loc="upper right", frameon=False)
        ax[r].locator_params(axis="x", nbins=10)
        ax[r].locator_params(axis="y", nbins=11)
        ax[r].set_xlim(left=float(X[0]), right=float(X[-1]))

    fig.suptitle(titles[0], fontsize=10)
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    figure = Image.open(buffer)
    plt.close("all")
    return figure
