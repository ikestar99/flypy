#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 03:38:05 2021

@author: ike
"""


import io
import time
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


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


def formatTitle(conserved, label):
    if conserved in label:
        label = label.replace(conserved, "")

    label = label.strip()
    return label


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


def lineGraph(Ys, titles, dYs=None, light=None, subs=[""]):
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
    # sns.set_style("darkgrid")
    fig, ax = plt.subplots(
        nrows=len(subs), figsize=(4, (4 * len(subs))), sharex=True, dpi=150)
    ax = ([ax] if len(subs) == 1 else ax)
    # n_colors = Ys.shape[1] // len(subs)
    Ys = Ys.reindex(sorted(Ys.columns), axis=1)
    columns = Ys.columns.values.tolist()
    X = Ys.index.values
    for r, subKey in enumerate(subs):
        sns.despine(ax=ax[r], top=True, right=True, left=False, bottom=True)
        subset = [c for c in columns if subKey in c]
        newCols = {c: formatTitle(titles[0], c) for c in subset}
        Ys[subset].copy().rename(columns=newCols).plot(
            kind="line", ax=ax[r], xlabel=titles[1], ylabel=titles[2],
            colormap="gist_rainbow", sort_columns=True)
        if dYs is not None:
            for c in [c for c in dYs.columns.values.tolist() if c in subset]:
                ax[r].fill_between(
                    X, Ys[c] - dYs[c], Ys[c] + dYs[c], linewidth=0, alpha=0.5,
                    color=ax[r].lines[subset.index(c)].get_color())

        y_max = np.abs(ax[r].get_ylim()).max()
        # ax[r].vlines(x=0, ymin=-y_max, ymax=y_max, lw=1, color='black')
        ax[r].hlines(y=0, xmin=X[0], xmax=X[-1], lw=1, color='black')
        if light is not None:
            ax[r].axvspan(light[0], light[1], color="blue", lw=0, alpha=0.1)

        # ax[r].set_xlabel(titles[1])
        ax[r].set_ylabel(titles[2])
        ax[r].locator_params(axis="x", nbins=10)
        ax[r].locator_params(axis="y", nbins=9)
        ax[r].set_xlim(left=float(X[0]), right=float(X[-1]))
        ax[r].set_ylim(bottom=-y_max, top=y_max)
        ax[r].legend(
            loc="upper right", frameon=False, bbox_to_anchor=(1, 1))

    fig.suptitle(titles[0], fontsize=10)
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    figure = Image.open(buffer)
    plt.close("all")
    return figure


def boxPlot(
        data, catCol, valCol, title, rowCol=None, hueCol=None, bootstrap=1000,
        ci=0.95):
    plt.rcParams["font.size"] = "10"
    height = data[catCol].unique().size * 1
    height = (
        height * data[hueCol].unique().size if hueCol is not None else height)
    fg = sns.catplot(
        x=valCol, y=catCol, hue=hueCol, row=rowCol, data=data, orient="h",
        height=height, aspect=float(5 / height), palette="flare",
        order=sorted(data[catCol].unique().flatten().tolist()),
        hue_order=(
            sorted(data[hueCol].unique().flatten().tolist())
            if hueCol is not None else None),
        row_order=(
            sorted(data[rowCol].unique().flatten().tolist())
            if rowCol is not None else None),
        legend=True, legend_out=True, facet_kws=dict(despine=True),
        kind="bar", capsize=.2, ci=ci, n_boot=bootstrap, sharex=False)

    for ax in fg.axes.flatten():

        ax.set_yticklabels(
            ax.get_yticklabels(), rotation=45, verticalalignment='top')
        ax.locator_params(axis="x", nbins=10)
        title = ax.get_title()
        ax.set_xlabel(title)
        ax.set_title("")
        sns.despine(ax=ax, top=True, right=True, left=True, bottom=False)
        ax.axvline(x=0, lw=2, color='black')
        ax.set_xlim(left=min(ax.get_xlim()), right=max(ax.get_xlim()) * 1.25)

    fig = plt.gcf()
    fig.suptitle(title, fontsize=10)
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    figure = Image.open(buffer)
    plt.close("all")
    return figure
