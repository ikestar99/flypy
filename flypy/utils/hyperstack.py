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


def _create_hyperstack(
        array: np.ndarray,
        shape: str
):
    """
    Convert an arbitrarily shaped image array into a uniform 5D hyperstack.

    Args:
        array (numpy.ndarray): Input array of images to be converted into a
            hyperstack.
        shape (str): Axis order of the array. It consists exclusively of
            letters:
            T: Time, time series.
            Z: Depth, z-slices.
            C: Channels, multi-color images.
            Y: Height, vertical pixel position.
            X: Width, horizontal pixel position.

            Example: "CTXY" --> array.ndim = 4, array.shape[1] = length of time
            series.

    Returns:
        numpy.ndarray: Input array reorganized to shape = "TZCYX", where:
            array.ndim = 5 and array.shape[2] = the number of channels per
            image.

    Raises:
        ValueError: If the input array and shape include a dimension other
            than the expected T, Z, C, X, or Y.

    Example:
        >>> test_array = np.random.rand(3, 5, 4, 2)  # Shape: (2, 5, 4, 3)
        >>> test_shape = "CXYZ"
        >>> test_hyperstack = _create_hyperstack(test_array, test_shape)
        >>> test_hyperstack.shape
        (1, 2, 3, 4, 5)

    Error Example:
        >>> test_shape = "AXYZ"
        >>> test_hyperstack = _create_hyperstack(test_array, test_shape)
        Traceback (most recent call last):
            ...
        ValueError: Input array includes unexpected dimensions.

    """
    expected_shape = "TZCYX"
    check = [d in expected_shape for d in shape]
    if not all(check):
        raise ValueError("Input array includes unexpected dimensions.")

    for i, d in enumerate(expected_shape):
        if d not in shape:
            # Add a new axis of length 1 at the appropriate position and
            # update shape
            shape = shape[:i] + d + shape[i:]
            array = np.expand_dims(array, axis=i)
        elif shape.index(d) != i:
            # Move d to the correct index (i) and update shape
            pos = shape.index(d)
            array = np.moveaxis(array, source=pos, destination=i)
            shape = shape.replace(d, "")
            shape = shape[:i] + d + shape[i:]

    return array


def _save_pil_array(
        file: str,
        image_list: list,
        squeeze: str = "tiff_deflate"
):
    """
    Save a list of Pillow images as a multipage TIFF.

    Args:
        file (str): Path to save the multipage TIFF file.
        image_list (list): List of Pillow images.
        squeeze (str, optional): Compression algorithm for saving  PIL images.
            Passed as an argument to the function call for PIL.Image.save.
            Defaults to "tiff_deflate".

    Raises:
        ValueError: If the file_path input is not a .tif path.
        TypeError: If image_list is not an iterabl.

    Example:
        >>> test_path = "path/to/multipage.tif"
        >>> image1 = Image.new("RGB", (256, 256), color="red")
        >>> image2 = Image.new("RGB", (256, 256), color="green")
        >>> image3 = Image.new("RGB", (256, 256), color="blue")
        >>> test_list = [image1, image2, image3]
        >>> _save_pil_array(test_path, test_list)

    Error Examples:
        # File input is not a .tif path
        >>> invalid_file = "path/to/multipage.png"
        >>> _save_pil_array(invalid_file, test_list)
        Traceback (most recent call last):
            ...
        ValueError: File input must be a .tif path.

        # image_list is not a list of PIL Images
        >>> invalid_image_list = [1, 2, 3]
        >>> _save_pil_array(test_path, invalid_image_list)
        Traceback (most recent call last):
            ...
        TypeError: image_list must be a list of PIL images.
    """
    if not file.endswith(".tif"):
        raise ValueError("File input must be a .tif path.")

    check = [type(i) == Image.Image for i in image_list]
    if not all(check):
        raise TypeError("image_list must be a list of PIL images.")

    image_list[0].save(
        file, compression=squeeze, save_all=True, append_images=image_list[1:])


def load_hyperstack(
        file: str,
        shape: str = ""
):
    """
    Load hyperstack from image.tif file usinf tifffile backend.

    Args:
        file (str): Path to .tif file to be loaded.
        shape (str, optional): Dimension order of hyperstack, if known. Passed
            to _create_hyperstack function call. Valid entries are combinations
            of "T", "Z", "C", "Y", "X". See _create_hyperstack for more
            information. If not provided, shape inferred from image metadata
            using tifffile backend. Defaults to "".

    Returns:
        numpy.ndarray: TIFF image converted to a 5D hyperstack array.

    Raises:
        ValueError: loaded image has a different number of dimensions than
            implied by shape parameter.

    Example:
        >>> test_filepath = "path/to/tiff/image.tif" # 3D image, YXC
        >>> test_hyperstack = load_hyperstack(test_filepath)
        >>> print(type(test_hyperstack), test_hyperstack.ndim)
        <class 'numpy.ndarray'>, 5

    Error Example:
        >>> test_hyperstack = load_hyperstack(test_filepath, shape="YX")
        Traceback (most recent call last):
            ...
        ValueError: Image has a different number of dimensions than shape.
    """
    with tf.TiffFile(file) as tif:
        # Load TIFF as numpy array
        array = tif.asarray()
        # Extract TIFF dimensions and dimension order as a string from metadata
        shape = (tif.series[0].axes if shape == "" else shape)

    if array.ndim != len(shape):
        raise ValueError(
            "Image has a different number of dimensions than shape.")

    array = _create_hyperstack(array, shape)
    return array


def load_stack(
        file: str
):
    """
    Load multipage image file as a numpy array.

    Note:
        This function fails to account for multidimensional data (e.g.,
        time series of z stacks) and is considered a less desirable method than
        load_hyperstack for loading all varieties of TIFF images. However,
        loadStack uses Pillow and is more likely to work with non-TIFF
        multipage images. The number and order of dimensions in the array
        depend on the input filetype.

    Args:
        file (str): Path to .tif file to be loaded.

    Returns:
        numpy.ndarray: Multipage image converted to a 3D multipage array.

    Warning:
        The number, length, and order of dimensions in the output array depend
        on the filetype of the input image. It is essential to normalize these
        parameters before using other hyperstack functions.

    Example:
        >>> test_filepath = "path/to/multipage/image.tif"
        >>> test_stack = load_stack(test_filepath)
        >>> print(type(test_stack))
        <class 'numpy.ndarray'>
    """
    array = Image.open(file)
    array = np.array([np.array(i) for i in ImageSequence.Iterator(array)])
    return array


def save_hyperstack(
        file: str,
        array: np.ndarray,
        dtype: np.dtype = np.uint8
):
    """
    Save hyperstack as image.tif.

    Args:
        file (str): Complete file path to .tif file save location.
        array (numpy.ndarray): Input array of images to be saved, hyperstack
            or otherwise.
        dtype (numpy.dtype, optional): Data type of the image array to be
            saved.
            Defaults to np.uint8.

    Raises:
        ValueError: If the input array is not a 5D hyperstack numpy array.
        ValueError: If the file input is not a .tif path.

    Example:
        >>> test_file = "path/to/image.tif"
        >>> test_hyperstack = np.zeros((10, 3, 256, 256), dtype=np.uint16)
        >>> save_hyperstack(test_file, test_hyperstack)

    Error Examples:
        # Input array is not a 5D hyperstack numpy array
        >>> invalid_array = np.zeros((256, 256), dtype=np.uint8)
        >>> save_hyperstack(test_file, invalid_array)
        Traceback (most recent call last):
            ...
        ValueError: Input array must be a 5D hyperstack numpy array.

        # File input is not a .tif path
        >>> invalid_file = "path/to/image.png"
        >>> save_hyperstack(invalid_file, test_hyperstack)
        Traceback (most recent call last):
            ...
        ValueError: File input must be a .tif path.
    """
    if array.ndim != 5:
        raise ValueError("Input array must be a 5D hyperstack numpy array.")

    if not file.endswith(".tif"):
        raise ValueError("File input must be a .tif path.")

    array = array.astype(dtype=dtype)
    tf.imwrite(file=file, data=array, imagej=True)


def save_stack(
        file: str,
        array: np.ndarray,
        mode: str = "L"
):
    """
    Save 3D image stack as a compressed multipage tiff.

    Args:
        file (str): Complete file path to .tif file save location.
        array (numpy.ndarray): 3D input array of images to be saved.
        mode (str, optional): Mode for converting 2D array into a PIL image.
            Passed as an argument to the function call for PIL.Image.fromarray.
            Defaults to "L".

    Raises:
        ValueError: If the file input is not a .tif path.

    Example:
        >>> test_file = "path/to/stack.tif"
        >>> test_stack = np.random.randint(0, 255, size=(10, 256, 256))
        >>> save_stack(test_file, test_stack)

    Error Example:
        # File input is not a .tif path
        >>> invalid_file = "path/to/stack.png"
        >>> save_stack(invalid_file, test_stack)
        Traceback (most recent call last):
            ...
        ValueError: File input must be a .tif path.
    """
    if not file.endswith(".tif"):
        raise ValueError("File input must be a .tif path.")

    # Flatten all leading axes to form a 3D image array
    array = np.reshape(array, (-1, array.shape[-2], array.shape[-1]))
    # Convert 3D array to 1D PIL image list
    array = [
        Image.fromarray(array[x], mode=mode) for x in range(array.shape[0])]
    # Save image list in a single tiff
    _save_pil_array(file, array)


def concatenate_hyperstacks(
        arrays: list,
        axis: int
):
    """
    Concatenate multiple hyperstacks of identical shape in all axes except for
    the concatenation axis.

    Args:
        arrays (list): List of 5D hyperstacks with identical shapes except for
            the axis along which to concatenate. The concatenation axis must
            correspond to the "axis" parameter.
        axis (int): Axis along which to do concatenation:
            0: concatenate first time dimension
            1: concatenate second depth dimension
            2: concatenate third channel dimension

    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Concatenated hyperstack of the same shape as the
                input arrays except along the concatenation axis.
            - list: List of ordered lengths of input arrays along the
                concatenation axis.

    Raises:
        TypeError: If any element in the list is not a numpy array.
        AssertionError: If the input arrays have different lengths along axes
            other than the concatenation axis.
        ValueError: If any input array is not 5D.

    Example:
        >>> array1 = np.ones((2, 3, 4, 5, 6))
        >>> array2 = np.ones((2, 5, 4, 5, 6))
        >>> array3 = np.ones((2, 1, 4, 5, 6))  # Different length along axis 1
        >>> test_list = [array1, array2, array3]
        >>> test_array, test_size = concatenate_hyperstacks(test_list, 1)
        >>> print(test_array.shape, test_size)
        (2, 9, 4, 5, 6) [3, 5, 1]

    Error Examples:
        # Non-array element in the list
        >>> invalid_list = [array1, array2, "not an array"]
        >>> test_array, test_size = concatenate_hyperstacks(invalid_list, 1)
        Traceback (most recent call last):
            ...
        TypeError: All elements in the list must be numpy arrays.

        # Arrays with different lengths along the concatenation axis
        >>> invalid_array = np.ones((2, 1, 4, 7, 6))
        >>> test_list = [array1, array2, invalid_array]
        >>> test_array, test_size = concatenate_hyperstacks(test_list, 1)
        Traceback (most recent call last):
            ...
        AssertionError: Arrays must have the same length along non-axis axes.

        # Non-hyperstack in the list
        >>> invalid_array = np.ones((2, 1, 4, 6))
        >>> test_list = [array1, array2, invalid_array]
        >>> test_array, test_size = concatenate_hyperstacks(test_list, 1)
        Traceback (most recent call last):
            ...
        ValueError: Input arrays must be 5D hyperstacks.
    """
    check = [type(a) == np.ndarray for a in arrays]
    if not all(check):
        raise TypeError("All elements in the list must be numpy arrays.")

    check = [a.ndim == 5 for a in arrays]
    if not all(check):
        raise ValueError("Input arrays must be 5D hyperstacks.")

    check = [list(a.shape) for a in arrays]
    [check[i].pop(axis) for i in range(len(check))]
    assert check.count(check[0]) == len(check), (
        "Arrays must have the same length along non-axis axes.")

    sizes = [array.shape[axis] for array in arrays]
    array = np.concatenate(arrays, axis=axis)
    return array, sizes


def splitTZCYXArray(
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

    Examples:
        >>> test_array = np.ones((2, 3, 4, 5, 6))
        >>> test_lengths = [1, 2]
        >>> test_arrays = splitTZCYXArray(test_array, test_lengths, 1)
        >>> print([a.shape for a in test_arrays])
        [(2, 1, 4, 5, 6), (2, 2, 4, 5, 6)]

    Error Examples:
        # lengths includes non integer entry
        >>> invalid_lengths = [1, "not an integer"]
        >>> test_arrays = splitTZCYXArray(test_array, invalid_lengths, 1)
        Traceback (most recent call last):
            ...
        TypeError: All elements in lengths list must be integers.

        Sum of lengths exceeds the array shape along the split axis
        >>> invalid_lengths = [5, 6, 7, 8]
        >>> test_arrays = splitTZCYXArray(test_array, invalid_lengths, 1)
        Traceback (most recent call last):
            ...
        ValueError: sum(lengths) exceeds the length of the split axis.
    """
    check = [isinstance(x, int) for x in lengths]
    if not all(check):
        raise TypeError("All elements in lengths list must be integers.")

    if sum(lengths) > array.shape[axis]:
        raise ValueError("sum(lengths) exceeds the length of the split axis.")

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

    Example:
        >>> test_array = np.array([[1, 2, 3, 4],[5, 6, 7, 8]])
        >>> test_block = (2, 2)
        >>> downsampled_array = downsample(test_array, test_block)
        >>> print(downsampled_array)
        [[3.5 5.5]]

    Error Examples:
        # Invalid block size
        >>> invalid_block = (5,)
        >>> downsampled_array = downsample(test_array, invalid_block)
        Traceback (most recent call last):
            ...
        ValueError: Block input must specify a scalar for each array dimension.

        # Invalid block size
        >>> invalid_block = (5, "b")
        >>> downsampled_array = downsample(test_array, invalid_block)
        Traceback (most recent call last):
            ...
        TypeError: All elements in block must be integers.

        # Block size larger than array dimensions
        >>> invalid_block = (5, 5)
        >>> downsampled_array = downsample(test_array, invalid_block)
        Traceback (most recent call last):
            ...
        IndexError: Downsample block size must be smaller than array.
    """
    if len(array.shape) != len(block):
        raise ValueError(
            "Block input must specify a scalar for each array dimension.")

    check = [isinstance(x, int) for x in block]
    if not all(check):
        raise TypeError("All elements in block must be integers.")

    check = [a >= b for a, b in zip(array.shape, block)]
    if not all(check):
        raise IndexError("Downsample block size must be smaller than array.")

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
#     pass
