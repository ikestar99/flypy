#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:06:21 2021
Core code obtained from Vicky Mak, Barnhart Lab
Class structure and relative imports from Ike Ogbonna, Barnhart Lab
@author: ike
"""

import numpy as np
import skimage.util as sku
import skimage.transform as skt

from pystackreg import StackReg


tReg = dict(
    translation=StackReg(StackReg.TRANSLATION),
    rigid=StackReg(StackReg.RIGID_BODY),
    rotation=StackReg(StackReg.SCALED_ROTATION),
    affine=StackReg(StackReg.AFFINE),
    bilinear=StackReg(StackReg.BILINEAR))


def registerImage(ref, mov, mode="rigid"):
    """
    Register image to a static reference using constrained transformations

    @param ref: 2D static reference image
    @type ref: numpy.ndarray
    @param mov: 2D image to be mapped onto ref
    @type mov: numpy.ndarray
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string

    @return tmat: 2D transformation matrix mapping mov onto ref
    @rtype tmat: numpy.ndarray
    """
    tmat = tReg[mode].register(ref, mov)
    return tmat


def registerStack(ref, mov, mode="rigid"):
    """
    Register stack to a static reference using constrained transformations

    @param ref: 2D static reference image
    @type ref: numpy.ndarray
    @param mov: 3D stack to be mapped onto ref
    @type mov: numpy.ndarray
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string

    @return tmats: 2D transformation matrices mapping mov onto ref[idx]
    @rtype tmats: list
    """
    tmats = [tReg[mode].register(ref, mov[x]) for x in range(mov.shape[0])]
    return tmats


def transformImage(mov, tmat):
    """
    Transform image using a known transformation matrix

    @param mov: 2D image to be transformed
    @type mov: numpy.ndarray
    @param tmat: 2D transformation matrix with which to transform mov
    @type tmat:

    @return moved: transformed 2D image
    @rtype moved: numpy.ndarray
    """
    # transform image with nearest neighbor interpolation using skimage
    moved = skt.warp(mov, tmat, order=0, mode="constant", cval=0)
    # convert foating-point image to an unsigned 8-bit image
    moved = sku.img_as_ubyte(moved)
    return moved


def transformStack(mov, tmats):
    """
    Transform stack using a known transformation matrix

    @param mov: 3D stack to be transformed
    @type mov: numpy.ndarray
    @param tmats: 2D transformation matrices with which to transform mov
    @type tmats: list

    @return moved: transformed 3D stack
    @rtype moved: numpy.ndarray
    """
    moved = [transformImage(mov[x], tmats[x]) for x in range(mov.shape[0])]
    moved = np.array(moved)
    return moved


def alignStack(ref, mov, mode="rigid"):
    """
    Register and transform an entire stack with a specific reference image

    @param ref: 2D static reference image
    @type ref: numpy.ndarray
    @param mov: 3D stack to be mapped onto ref
    @type mov: numpy.ndarray
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string

    @return moved: aligned hyperstack of the same shape as mov
    @rtype moved: numpy.ndarray
    @return tmats: 2D transformation matrices mapping mov onto ref[idx]
    @rtype tmats: list
    """
    tmats = registerStack(ref, mov, mode)
    moved = transformStack(mov, tmats)
    return moved, tmats


def alignTHyperstack(ref, mov, channel, mode="rigid"):
    """
    Register and transform T of an entire hyperstack with a specific reference
    hyperstack of identical shape in all axes except for T

    @param ref: 5D static reference hyperstack with identical shape to mov,
        except for an axis of length 1 corresponding to the T axis
    @type ref: numpy.ndarray
    @param mov: 5D hyperstack to be transformed
    @type mov: numpy.ndarray
    @param channel: reference channel from which to derive transformation
        matrices
    @type channel: int
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string

    @return moved: aligned hyperstack of the same shape as mov
    @rtype moved: numpy.ndarray
    """
    moved = np.zeros(mov.shape)
    for z in range(mov.shape[1]):
        tmats = registerStack(
            ref[0, z, channel], mov[:, z, channel], mode=mode)
        for c in range(mov.shape[2]):
            moved[:, z, c] = transformStack(mov[:, z, c], tmats)

    return moved


def alignZHyperstack(ref, mov, channel, mode="rigid"):
    """
    Register and transform Z of an entire hyperstack with a specific reference
    hyperstack of identical shape in all axes except for Z

    @param ref: 5D static reference hyperstack with identical shape to mov,
        except for an axis of length 1 corresponding to the Z axis
    @type ref: numpy.ndarray
    @param mov: 5D hyperstack to be transformed
    @type mov: numpy.ndarray
    @param channel: reference channel from which to derive transformation
        matrices
    @type channel: int
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string

    @return moved: aligned hyperstack of the same shape as mov
    @rtype moved: numpy.ndarray
    """
    moved = np.zeros(mov.shape)
    for t in range(mov.shape[0]):
        tmats = registerStack(
            ref[t, 0, channel], mov[t, :, channel], mode=mode)
        for c in range(mov.shape[2]):
            moved[t, :, c] = transformStack(mov[t, :, c], tmats)

    return moved


def alignHyperstack(mov, channel, mode):
    """
    Register and transform an entire hyperstack

    @param mov: 5D hyperstack to be transformed
    @type mov: numpy.ndarray
    @param channel: reference channel on which to conduct transformation
    @type channel: int
    @param mode: contraints of registration paradigm:
        translation: translation in X/Y directions
        rigid: rigid transformations
        rotation: rotation and dilation
        affine: affine transformation
        bilinear: bilinear transformation
    @type mode: string

    @return moved: aligned hyperstack of the same shape as mov
    @rtype moved: numpy.ndarray
    """
    moved = None
    if mov.shape[0] > 1:
        ref = np.mean(mov, axis=0, keepdims=True).astype("uint8")
        moved = alignTHyperstack(ref, mov, channel=channel, mode=mode)
    if mov.shape[1] > 1:
        ref = np.mean(mov, axis=1, keepdims=True).astype("uint8")
        moved = alignZHyperstack(ref, mov, channel=channel, mode=mode)

    return moved
