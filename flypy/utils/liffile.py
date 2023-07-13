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


def load_lif_file(
        path: str
):
    """
    Load a Leica .lif file as an object from which to extract imaging samples.

    Args:
        path (str): Path to .lif file to be loaded.

    Returns:
        LifFile: LifFile iterable that contains all imaged samples in the lif
            file.

    Raises:
        ValueError: path is not a .lif file

    Example;
        >>> test_path = "path/to/file.lif"
        >>> test_lif = load_lif_file(test_path)

    Error Example:
        >>> invalid_path = "path/to/file.lif"
        >>> load_lif_file(invalid_path)
        Traceback (most recent line last):
            ...
        ValueError: Path must be a .lif file.
    """
    if not path.endswith(".lif"):
        raise ValueError("Path must be a .lif file.")

    lif = LifFile(path)
    return lif


def getLifImage(
        lif: LifFile,
        idx: int,
        dtype: np.dtype = np.uint8
):
    """
    Extract an imaged sample as a hyperstack from a pre-loaded .lif file.

    Args:
        lif (LifFile): LifFile iterable that contains all imaged samples in the
            .lif file.
        idx (int): Index of the desired image within the lif file (0-indexed).
        dtype (np.dtype, optional): Data type of the image array to be saved.
            Passed to the np.ndarray.astype constructor. Defaults to np.uint8.

    Returns:
        numpy.ndarray: Lif imaging sample converted to a 5D hyperstack array.
            Hyperstack dimension order is "TZCYX".
    """
    image = lif.get_image(img_n=idx)
    hyperstack = [[[np.array(image.get_frame(t=t, z=z, c=c))
                    for c in range(image.channels)]
                   for z in range(image.dims.z)]
                  for t in range(image.dims.t)]
    hyperstack = np.array(hyperstack, dtype=dtype)
    return hyperstack
