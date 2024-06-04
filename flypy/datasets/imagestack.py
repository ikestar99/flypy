#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import numpy as np
import tifffile as tf
import skimage.measure as skm

from PIL import Image, ImageSequence


"""
Helper functions for loading and modifying imaging data as hyperstacks.
Hyperstack axis order is TZCYX
"""


class ImageStack:
    """
    Class to store and manipulate imaging data in hyperstack form.

    Imaging data is stored in a 5D numpy array while corresponding axis labels
    are stored in a list.

    Attributes:
        _SHAPE (str): Standard hyperstack axis order.
        frames (np.ndarray): Image stack.
        shape (str): Corresponding axis order for image stack.
        labels (np.list): Labels of each axis in image stack.
    """
    _SHAPE = "TZCYX"

    def __init__(
            self,
            frames: np.ndarray,
            shape: str,
            labels: dict = None
    ):
        """
        Instantiate ImageStack dataset.

        NOTE: Imaging dataset in frames arg will be reordered and reshaped to
        match the standard hyperstack axis order. shape arg will be reorganized
        in tandem, without supplementing omitted axes.

        Args:
            frames (numpy.ndarray): Array of underlying imaging data. Must be
                <= 5D where each axis corresponds to an axis in class _SHAPE
                attribute.
            shape (str): Axis order of frames arg. Must be a whole or subset of
                the following letters:

                "T": Time, time series.
                "Z": Depth, z-slices.
                "C": Channels, multi-color images.
                "Y": Height, vertical pixel position.
                "X": Width, horizontal pixel position.

                Example: "CTXY" --> frames.ndim = 4, frame.shape[1] = length of
                time series. Arg used to reorder axes in frames arg to standard
                order in class _SHAPE attribute.
            labels (dict, optional): Labels along each axis in frames arg.
                key (str): Axis. Valid entries are "T", "Z", "C", "Y", "X".
                item (np.ndarray): Corresponding labels for each index along
                    specified axis.
                Axes omitted from labels arg are labeled numerically by index.
                Defaults to None, in which case all axes are labeled by index.
        """
        self.frames = frames.copy()
        self.shape = shape.upper().copy()
        self.labels = (dict() if labels is None else labels).copy()

        # reorder axes
        self._reorder_axes()

    def _reorder_axes(
            self,
    ):
        """
        Reorder axes of instance shape attribute to standard hyperstack order.
        Target order described by class _SHAPE attribute. Singleton axes are
        added as needed to reach 5D.
        """
        # determine current order of axes and add singletons if missing
        add_dims = [d for d in self._SHAPE if d not in self.shape]
        old_dims = list(self.shape) + add_dims
        new_image_idxs = [old_dims.index(d) for d in self._SHAPE]
        new_axis_idxs = [
            self.shape.index(d) for d in self._SHAPE if d in self.shape]

        # update instance attributes to normative order defined by class _SHAPE
        self.frames = self.frames[
            tuple([Ellipsis] + [np.newaxis for _ in add_dims])]
        self.frames = np.transpose(self.frames, new_image_idxs)
        self.shape = "".join(self.shape[i] for i in new_axis_idxs)

    def _sort_axes(
            self
    ):
        """
        Sort each labeled axis in ascending label order.

        Returns:
            (ImageStack): New instance with labeled axes sorted.
        """
        labels = self.labels.copy()
        frames = self.frames.copy()
        for key, idx in {k: v for k, v in labels.items() if v.size > 1}:
            idx = np.argsort(idx)
            labels[key] = labels[key][idx]
            frames = np.take(frames, idx, axis=self._SHAPE.index(key))

        return ImageStack(frames, self.shape, labels)

    def __len__(
            self
    ):
        """
        Returns:
            (int): Total number of (T, Z, C) images stored.
        """
        return np.prod(self.frames.shape[:-2])

    def __getitem__(
            self,
            axe_dict: dict
    ):
        """
        Extract a subset of indices along specified axes.

        NOTE: items in axe_dict arg are used to index underlying array data
        directly. Valid types include int, list, boolean mask, etc. __getitem__
        does not support indexing by label value: use filter method instead.

        Args:
            axe_dict (dict): Indices to extract from each axis.
                key (str): Axis. Valid entries are "T", "Z", "C", "Y", "X".
                item: Corresponding indices to extract.

        Returns:
            (ImageStack): New instance with subset of data specified by
                axe_dict arg.
        """
        axe_dict = {k.upper(): v for k, v in axe_dict.items()}
        frames = self.frames.copy()
        labels = self.labels.copy()

        # filter frames and labels along each axis separately
        for key, idx in axe_dict.items():
            axis = self._SHAPE.index(key)
            frames = np.take(frames, idx, axis=axis)
            frames = (
                frames if frames.ndim == len(self._SHAPE)
                else np.expand_dims(frames, axis=axis))
            if key in labels:
                labels[key] = (
                    labels[key][idx] if type(labels[key][idx]) is np.ndarray
                    else np.array([labels[key][idx]]))

        # remove filler axes
        idx = [slice(None) if a in axe_dict else 0 for a in self._SHAPE]
        return ImageStack(frames[tuple(idx)], self.shape, labels)

    def filter(
            self,
            axe_dict: dict
    ):
        """
        Extract a subset of labels along specified axes. Effectively a wrapper
        for __getitem__ to extract specific labels rather than indices from
        specified axes.

        Args:
            axe_dict (dict): Labels to extract from each axis.
                key (str): Axis. Valid entries are "T", "Z", "C", "Y", "X".
                item (list): Corresponding labels to extract.

        Returns:
            (ImageStack): New instance with subset of data specified by
                axe_dict arg.
        """
        axe_dict = {
            k: np.in1d(self.labels[k], v) for k, v in axe_dict.items() if k in
            self.labels}
        return self[axe_dict]

    @classmethod
    def concatenate(
            cls,
            stacks: list,
            axis: str,
            pre_sort: bool = False
    ):
        """
        Concatenate multiple ImageStack instances along specified axis. Each
        frame attribute should be of identical shape in all axes except for
        the concatenation axis.

        WARNING: pre_sort arg should only be used if all ImageStack instances
        have the same set of labeled axes and labels within each axis.

        Args:
            stacks (list): List of ImageStack instances to concatenate.
            axis (str): Axis along which to concatenate. Valid entries include:
                "T": Temporal concatenation.
                "Z": Spatial concatenation along depth axis.
                "C": Channels concatenation.
                "Y": Spatial concatenation along vertical axis.
                "X": Spatial concatenation along horizontal axis.
            pre_sort (bool, optional). If True, sort all elements of stacks arg
                in ascending order within labeled axes before concatenating.
                Defaults to False, in which case non-concatenation axes retain
                their initial order.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: Concatenated hyperstack of the same shape as the
                    input arrays except along the concatenation axis.
                - list: List of ordered lengths of input arrays along the
                    concatenation axis.
        """
        stacks = ([s._sort_axes() for s in stacks] if pre_sort else stacks)
        frames = [s.frames for s in stacks]
        labels = stacks[0].labels.copy()
        frames = np.concatenate(frames, axis=cls._SHAPE.index(axis))
        if axis in labels:
            labels[axis] = np.concatenate([s.labels[axis] for s in stacks])

        return ImageStack(frames, stacks[0].shape, labels)

    @classmethod
    def from_tif(
            cls,
            file: str,
            labels: dict,
            shape: str = None
    ):
        """
        Instantiate ImageStack dataset directly from .tif file

        Args:
            file (str): Path to .tif file.
            labels (dict): Axes labels of each axis in loaded array. Passed to
                __init__.
            shape (str, optional): Dimension order of hyperstack, if known.
                Passed to __init__ function call. Valid entries are
                permutations of "TZCYX". See __init__ for more details.
                Defaults to None, in which case shape is inferred from image
                metadata using tifffile backend.

        Returns:
            (ImageStack): New instance with data in file arg.
        """
        with tf.TiffFile(file) as tif:
            # Load TIFF as numpy array, extract TIFF axis order as a string
            array = tif.asarray()
            shape = (tif.series[0].labels if shape is None else shape)

        return cls(array, shape, labels)

    @classmethod
    def from_pil(
            cls,
            file: str,
            shape: str,
            labels: dict,
            r_axis: int = 0,
            reshape: list = None
    ):
        """
        Instantiate ImageStack dataset directly from .tif file

        NOTE: This function fails to account for multidimensional data (e.g.,
        time series of z stacks) and is a less desirable method than from_tif
        for loading .tif hyperstack data. from_PIL uses Pillow backend and is
        more likely to work with non-TIFF multipage images.

        Warning:
        The number, length, and order of axes in Pillow-loaded array may vary
        on the filetype of the input image. Know which dimension is which
        based on the filetype and update shape arg accordingly. Alternatively,
        load and process hyperstack data elsewhere and call ImageStack __init__
        directly for more predictable behavior

        Args:
            file (str): Path to multipage image.
            shape (str): Dimension order of hyperstack, if known.
                Passed to __init__ function call. Valid entries are
                permutations of "TZCYX". See __init__ for more details.
            labels (dict): Axes labels of each axis in loaded array. Passed to
                __init__.
            r_axis (int, optional): Axis of image stack loaded from file arg to
                unpack if reshape arg specified. Defaults to 0.
            reshape (list, optional): Shape of axes to unpack from r_axis axis.
                Product of elements in reshape must equal array.shape[r_axis].
                Shape arg should include all unpacked axes in the order that
                they are unpacked, in the same position specified by r_axis
                arg.

        Returns:
            Returns:
                (ImageStack): New instance with data in file arg.
        """
        # load image data
        array = Image.open(file)
        array = np.array([np.array(i) for i in ImageSequence.Iterator(array)])

        # reshape image data
        new_dims = list(array.shape)
        new_dims[r_axis:r_axis + 1] = reshape
        array = array.reshape(*new_dims)
        return cls(array, shape, labels)

    @staticmethod
    def save_pil_list(
            file: str,
            image_list: list,
            squeeze: str = "tiff_deflate"
    ):
        """
        Save a list of Pillow images as a multipage .tif image.

        Args:
            file (str): Complete path to .tif save location.
            image_list (list): List of Pillow images.
            squeeze (str, optional): Compression algorithm for saving  PIL
                images. Passed to PIL.Image.save. Defaults to "tiff_deflate".
        """
        image_list[0].save(
            file, compression=squeeze, save_all=True,
            append_images=image_list[1:])

    def save_tif(
            self,
            file: str,
            dtype: np.dtype = np.uint8
    ):
        """
        Save instance frames attribute as .tif file using tifffile backend.

        Args:
            file (str): Complete path to .tif save location.
            dtype (numpy.dtype, optional): Data type of the image array to be
                saved. Defaults to np.uint8.
        """
        tf.imwrite(file, data=self.frames.astype(dtype=dtype), imagej=True)

    def save_pil(
            self,
            file: str,
            mode: str = "L"
    ):
        """
        Save instance frames attribute as .tif file using tifffile backend.

        Args:
            file (str): Complete path to .tif save location.
            mode (str, optional): Mode for converting 2D array into a PIL image.
                Passed as an argument to the function call for PIL.Image.fromarray.
                Defaults to "L".
        """
        # Flatten all leading axes to form a 3D image array
        frames = np.reshape(self.frames, (-1, *self.frames.shape[-2:]))

        # Convert 3D array to list of PIL images and save
        frames = [
            Image.fromarray(frames[x], mode=mode) for x in
            range(frames.shape[0])]
        self.save_pil_list(file, frames)

    def segment_hyperstack(
            array: np.ndarray,
            lengths: list,
            axis: int
    ):
        """
        Split a hyperstack into multiple hyperstacks of identical shape in all axes
        except for the split axis.

        Args:
            array (numpy.ndarray): 5D hyperstack to split.
            lengths (list): List of ordered lengths of output arrays along the
                split axis.
            axis (int): Axis along which to split:
                0: split the first time dimension.
                1: split across the second depth dimension.
                2: split across the third channel dimension.

        Returns:
            list: List of 5D hyperstacks with identical shapes except for the split
                axis. The shape of each array in the list is equal to the input
                array except for the length along the split axis, which corresponds
                to the corresponding length in the lengths input.

        Raises:
            TypeError: If lengths included non-integer types.
            ValueError: If the sum of lengths is greater than array.shape[axis].
        """
        # create slice objects along split axis to index desired subarrays
        lengths = lengths + [0]
        lengths = [sum(lengths[:x]) for x in range(len(lengths))]
        arrays = []
        for x in range(len(lengths) - 1):
            # extract sliced subarray and append to list of arrays
            arrays.append(np.take(
                array, tuple(range(lengths[x], lengths[x + 1])), axis=axis))

        return arrays

    def downsample(
            array: np.ndarray,
            block: tuple,
            func: np.ufunc = np.mean
    ):
        """
        Downsample an array by applying a function to non-overlapping blocks of the
        specified size along each dimension.

        Args:
            array (numpy.ndarray): Input array to be downsampled.
            block (tuple): Tuple specifying the block size for each dimension. The
                length of the tuple must be equal to the number of dimensions in
                the array. Each value in the tuple represents the scale factor
                along the corresponding dimension.
            func (numpy.ufunc, optional): Function to be applied to the blocks
                during downsampling. Defaults to numpy.mean.

        Returns:
            numpy.ndarray: Downsampled array obtained by applying the specified
                function to non-overlapping blocks of the specified size along each
                dimension.

        Raises:
            ValueError: If the length of the block tuple does not match the number
                of dimensions in the array.
            TypeError: If block includes non-integer types.
            IndexError: If any dimension of the block size is larger than the
                corresponding dimension of the array.
        """
        # Resize array to be evenly divisible by the scale factor along each axis
        size = [slice(int((o // e) * e)) for o, e in zip(array.shape, block)]
        array = array[tuple(size)]

        # Downsample array
        array = skm.block_reduce(array, block, func=func)
        return array

    # def binStack(array, bins, id0=1):
    #     """
    #     Bin a hyperstack according to a known list of frames per bin
    #
    #     @param array: input array to be binned, hyperstack or otherwise.
    #         Binning occurs along the first axis
    #     @type array: numpy.ndarray
    #     @param bins: list of frames in each bin. Each index in bins is a
    #         sublist of all frames that should be averaged to yield the index-th
    #         frame in the binned image. ie the following list of lists:
    #         [[1, 5, 8, 9]
    #          [2, 3, 6, 10]
    #          [4, 7, 11]]
    #         indicates that there are 3 binned frames from an imaging array with
    #         11 unbinned frame. The first binned image in this example includes
    #         all frames in bins[0]: 1, 5, 8, and 9
    #     @type bins: list
    #     @param id0: index of first frame according to frame numbering in bins
    #         id0=0 indicates frames are 0-indexes (first frame at index 0)
    #         id0=1 indicates frames are 1-indexed (first frame at ndex 1)
    #     @type id0: int
    #
    #     @return: binned image of same shape as input array except in axis 0
    #     @rtype: numpy.ndarray
    #     """
    #     # form a list of binned images with list comprehension
    #     bins = [set([min(x, array.shape[0] - id0) for x in b]) for b in bins]
    #     binned = [np.mean(np.array(
    #         [array[t] for t in bin]), axis=0) for bin in bins]
    #     binned = np.array(binned, dtype=array.dtype)
    #     return binned
