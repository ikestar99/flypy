#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

from PIL import Image, ImageSequence
import numpy as np
import tifffile as tf
import skimage.util as sku
import skimage.transform as skt

from pystackreg import StackReg
from readlif.reader import LifFile


"""
Collection of functions for creating and modifying hyperstacks of imaging data.
Hyperstack axis order is TZCYX
"""


tReg = dict(
    translation=StackReg(StackReg.TRANSLATION),
    rigid=StackReg(StackReg.RIGID_BODY),
    rotation=StackReg(StackReg.SCALED_ROTATION),
    affine=StackReg(StackReg.AFFINE),
    bilinear=StackReg(StackReg.BILINEAR))


def _getTZCYXArray(array, shape):
    """
    Convert arbitrarily shaped image array into uniform 5D hyperstack

    @param array: input array of images to be converted into a hyperstack
    @type array: numpy.ndarray
    @param shape: axis order of array, made exclusively of letters:
        T: time, time series
        Z: depth, z-slices
        C: channels, multi-color images
        Y: height, vertical pixel position
        X: width, horizontal pixel position

        "CTXY" --> array.ndim = 4, array.shape[1] = length of time series
    @type shape: string

    @return: input array reorganized to shape = "TZCYX", ie:
        array.ndim = 5, array.shape[2] = number of channels per image
    @rtype: numpy.ndarray
    """
    for i, d in enumerate("TZCYX"):
        if d not in shape:
            """
            if axis d is not in image, add a new axis of length 1 at the
            appropriate position and update shape
            """
            shape = "".join([shape[:i], d, shape[i:]])
            array = np.expand_dims(array, axis=i)
        elif shape.index(d) != i:
            """
            if axis d is in image but at the wrong index, move d to the correct
            index (i) and update shape
            """
            pos = shape.index(d)
            array = np.moveaxis(array, source=pos, destination=i)
            shape = shape.replace(d, "")
            shape = "".join([shape[:i], d, shape[i:]])

    return array


def loadTZCYXTiff(file):
    """
    Load hyperstack from image.tif file

    @param file: path to .tif file to be loaded
    @type file: string

    @return: tif image converted to 5D hyperstack array
    @rtype: numpy.ndarray
    """
    with tf.TiffFile(file) as tif:
        # load tif as numpy array
        array = tif.asarray()
        # extract tif dimensions and dimension order as string from metadata
        shape = tif.series[0].axes

    array = _getTZCYXArray(array, shape)
    return array


def loadMultipageTiff(file):
    """
    Load multipage image.tif file as numpy array

    @param file: path to .tif file to be loaded
    @type file: string

    @return: tif image converted to 3D multipage array
    @rtype: numpy.ndarray
    """
    array = Image.open(file)
    array = np.array([np.array(i) for i in ImageSequence.Iterator(array)])
    return array


def saveTZCYXTiff(file, array, shape, dtype=np.uint8):
    """
    Save hyperstack as image.tif

    @param file: complete file path to .tif file save location
    @type file: string
    @param array: input array of images to be saved, hyperstack or otherwise
    @type array: numpy.ndarray
    @param shape: axis order of array, made exclusively of letters:
        T: time, time series
        Z: depth, z-slices
        C: channels, multi-color images
        Y: height, vertical pixel position
        X: width, horizontal pixel position
    @type shape: string
    @param dtype: data type of image array to be saved
    @type dtype: np.dtype, passed to np.ndarray.astype constructor
    """
    array = _getTZCYXArray(array, shape).astype(dtype=dtype)
    tf.imwrite(file=file, data=array, imagej=True)


def savePillowArray(saveFile, pilArray):
    pilArray[0].save(
        saveFile, compression="tiff_deflate",
        save_all=True, append_images=pilArray[1:])


def saveMultipageTiff(file, array, mode="L"):
    """
    Save 3D image stack as a compressed multipage tiff

    @param file: complete file path to .tif file save location
    @type file: string
    @param array: 3D input array of images to be saved
    @type array: numpy.ndarray
    @param mode: mode for converting 2D array into a PIL image. Passed as
        argument to function call for PIL.Image.fromarray
    @type mode: string
    """
    # flatten all leading axes to form a 3D image array
    array = np.reshape(array, (-1, array.shape[-2], array.shape[-1]))
    # convert 3D array to 1D PIL image list
    array = [
        Image.fromarray(array[x], mode=mode) for x in range(array.shape[0])]
    # save image list in a single tif
    array[0].save(
        file, compression="tiff_deflate", save_all=True,
        append_images=array[1:])


def concatenateTZCYXArray(arrays, axis):
    """
    Concatenate multiple hyperstacks of identical shape in all axes except for
    the concatenation axis

    @param arrays: iterable of 5D hyperstacks with identical shapes except for
        axis along which to concatenate. Concatenation axis must correspond to
        "axis" parameter
    @type arrays: iterable
    @param axis: axis along which to do concatenation:
        0: concatenate first time dimension
        1: concatenate second depth dimension
        2: concatenate third channel dimension
    @type axis: int

    @return: concatenated hyperstack of same shape as input arrays except
        along concatenation axis

        list of ordered lengths of input arrays arrays along concatenation axis
    @rtype: numpy.ndarray, list
    """
    lengths = [array.shape[axis] for array in arrays]
    array = np.concatenate(arrays, axis=axis)
    return array, lengths


def splitTZCYXArray(array, lengths, axis):
    """
    Split hyperstack into multiple hyperstacks of identical shape in all axes
    except for the split axis

    @param array: 5D hyperstack to split
    @type array: numpy.ndarray
    @param lengths: list of ordered lengths of of output arrays arrays along
        split axis
    @type lengths: list
    @param axis: axis along which to split:
        0: split first time dimension
        1: split across second depth dimension
        2: split across third channel dimension
    @type axis: int

    @return: list of 5D hyperstacks with identical shapes except for split
        axis. Array[x].shape[axis] = length[x]
    @rtype: list
    """
    lengths = lengths + [0]
    lengths = [sum(lengths[:x]) for x in range(len(lengths))]
    slices = [slice(None)] * array.ndim
    arrays = list()
    for x in range(len(lengths) - 1):
        currentSlice = slices
        currentSlice[axis] = slice(lengths[x], lengths[x + 1])
        arrays = arrays + [array[tuple(currentSlice)]]

    return arrays
