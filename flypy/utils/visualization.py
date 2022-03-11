#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 03:38:05 2021

@author: ike
"""


import io
import time
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


COLORS = ("#FF0000", "#FF8000", "#FFFF00", "00FF00", "00FFFF", "0000FF",
          "7f00ff", "FF00FF", "FF00F7", "808080")
SPACE = "    "
TRUNK = "│   "
SPLIT = "├── "
FINAL = "└── "
    

def wait(message, multiplier=0.25):
    print(message)
    time.sleep((1.0 * multiplier))
    
    
def getInput(message=""):
    wait(message)
    value = input(">> ")
    return value


# def getDirectoryTree(dirDict):
#     def listDirs(dictionary, array=None, depth=1):
#         array = (
#             np.array([[".    "]]).astype('U100') if array is None else array)
#         for i, folder in enumerate(dictionary):
#             array = np.concatenate((array, array[-1][np.newaxis]), axis=0)
#             array[-1] = SPACE
#             while depth + 1 > array.shape[-1]:
#                 array = np.concatenate(
#                     (array, array[:,-1][:,np.newaxis]), axis=-1)
#                 array[:,-1] = SPACE
#
#             array[-1,depth] = folder
#             if i + 1 == len(dictionary):
#                 array[-1,depth - 1] = FINAL
#
#             if type(dictionary[folder]) is dict:
#                 array = listDirs(
#                     dictionary[folder], array=array, depth=depth + 1)
#
#         return array
#
#     def rowToString(rowArray):
#         string = "".join([rowArray[x] for x in range(rowArray.size)])
#         while string[-1] == " ":
#             string = string[:-1]
#
#         return string
#
#     array = listDirs(dictionary=dirDict)
#     count = 0
#     while True:
#         reference = (
#             (array == FINAL).astype(int) + (array == TRUNK).astype(int) +
#             (array == SPLIT).astype(int))
#         check = np.argwhere(reference)
#         for x in range(check.shape[0]):
#             if array[check[x,0] - 1, check[x,1]] == SPACE:
#                 count += 1
#                 array[check[x,0] - 1, check[x,1]] = (
#                     TRUNK if array[check[x,0] - 1, check[x,1] + 1] in
#                     (SPACE, FINAL, TRUNK, SPLIT) else SPLIT)
#
#         if count == 0:
#             break
#         else:
#             count = 0
#
#     string = "\n".join([rowToString(array[x]) for x in range(array.shape[0])])
#     return string


# colors = ("#F1160C", "#0AC523", "#0F37FF", "#8F02C5")
#
def makeFigure(X, Ys, titles, dYs=None, light=None, sub=None):
    plt.rcParams["font.size"] = "8"
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(
        nrows=len(Ys), figsize=(4, (3 * len(Ys))), sharex=True, dpi=150)
    ax = ([ax] if len(Ys) == 1 else ax)
    col = sns.color_palette("rocket", n_colors=len(Ys))  # "viridis"
    for r, key in enumerate(Ys.keys()):
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
