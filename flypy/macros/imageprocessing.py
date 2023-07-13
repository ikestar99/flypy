#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue July  4 13:06:21 2023
@author: ike
"""


from ..utils.hyperstack import (
    load_hyperstack, concatenate_hyperstacks, downsample, save_hyperstack)


def downsample_and_compile(
        path_list,
        save_path,
        shape: str,
        block: tuple,
        func: np.ufunc,
        axis: int,
        dtype: np.dtype
):
    """

    Args:
        path_list ():
        save_path ():
        shape (str): Shape of all input arrays. Passed to load_hyperstack
            function call.
        block (tuple): Shape of structuring element of downsample operation.
            Passed to downsample function call.
        func (numpy.ufunc): Downsampling operation. Passed to downsample
            function call.
        axis (int): Axis along which to perform image compilation. Passed to
            concateante_hyperstacks function call.
        dtype (np.dtype): Data type of saved image. Passed to save_hyperstack
            function call.
    """
    data = [
        downsample(
            load_hyperstack(path, shape), block, func)
        for path in path_list]
    data, _ = concatenate_hyperstacks(data, axis=axis)
    save_hyperstack(save_path, data, dtype=dtype)
