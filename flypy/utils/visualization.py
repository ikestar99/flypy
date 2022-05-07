#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 03:38:05 2021

@author: ike
"""


import io
import numpy as np
import pandas as pd
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt


def formatTitle(conserved, label):
    if conserved in label:
        label = label.replace(conserved, "")

    label = label.strip()
    return label


def getColors(n, palatte="viridis"):
    return sns.color_palette(palatte, n_colors=n)


def getFigAndAxes(rows=1, cols=1, sharex=False, sharey=False, **kwargs):
    plt.rcParams["font.size"] = "10"
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, squeeze=False, figsize=(4 * cols, 4 * rows),
        dpi=150, sharex=sharex, sharey=sharey, **kwargs)
    fig.subplots_adjust(top=0.9)
    return fig, axes


def figToImage(title="", fig=None):
    fig = (plt.gcf() if fig is None else fig)
    plt.tight_layout()
    fig.suptitle(title, y=.95, fontsize=10)
    buffer = io.BytesIO()
    fig.savefig(buffer)
    buffer.seek(0)
    figure = Image.open(buffer)
    plt.close("all")
    return figure


def clearAx(ax):
    ax.axis("off")


def redrawAxis(ax, xaxis=False, yaxis=False, yintercept=0, xintercept=0, lw=2):
    sns.despine(ax=ax, top=True, right=True, left=True, bottom=True)
    (ax.axhline(y=yintercept, xmin=0, xmax=1, lw=lw, color="k")
     if xaxis else None)
    (ax.axvline(x=xintercept, ymin=0, ymax=1, lw=lw, color="k")
     if yaxis else None)


def moveAxis(ax, yintercept=0, xintercept=0, lw=2):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)
    ax.spines["bottom"].set_position(('data', yintercept))
    ax.spines["bottom"].set_linewidth(lw)
    ax.spines["left"].set_position(('data', xintercept))
    ax.spines["left"].set_linewidth(lw)


def resetLimits(ax, xs=None, ys=None):
    (ax.set_xlim(*[float(x) for x in xs]) if xs is not None else None)
    (ax.set_ylim(*[float(y) for y in ys]) if ys is not None else None)


def mirrorAxisLimits(ax, axis):
    value = np.abs(ax.get_ylim() if axis == "0" else ax.get_xlim()).max()
    (ax.set_ylim(bottom=-value, top=value) if axis == "0" else ax.set_xlim(
        left=-value, right=value))


def adjustNBins(ax, xbins=10, ybins=8):
    ax.locator_params(axis="x", nbins=xbins)
    ax.locator_params(axis="y", nbins=ybins)


def addHorizontalAxTitle(ax, title):
    ax.set_title(title)


def addVerticalAxTitle(ax, title):
    ax.text(
        1.1, 0.5, title, verticalalignment='center', rotation=270,
        transform=ax.transAxes)


def addAxisTitles(ax, xtitle=None, ytitle=None):
    (ax.set_xlabel(xtitle) if xtitle is not None else None)
    (ax.set_ylabel(ytitle) if ytitle is not None else None)


def addLegend(ax):
    ax.legend(loc="center left", frameon=False, bbox_to_anchor=(1, 1))


def shadeVerticalBox(ax, start, stop, alpha=0.05):
    ax.axvspan(start, stop, color="gray", lw=0, alpha=alpha)


def linePlot(
        ax, data, yCol, xCol=None, hCol=None, hueDict=None, ci=95, bootN=1000,
        color=None, lw=2, raw=False):
    sns.lineplot(
        x=xCol, y=yCol, hue=hCol, data=data, palette=hueDict, ci=ci,
        n_boot=bootN, ax=ax, color=color, linewidth=lw)
    if raw:
        sns.lineplot(
            x=xCol, y=yCol, hue=hCol, data=data, palette=hueDict, ci=ci,
            n_boot=bootN, ax=ax, color=color, linewidth=lw, estimator=None,
            alpha=0.2, legend=None)


def boxPlot(
        ax, data, yCol, cCol=None, hCol=None, hueDict=None, color=None, lw=2,
        raw=False, ori="v"):
    sns.boxplot(
        x=cCol, y=yCol, hue=hCol, data=data, palette=hueDict, ax=ax,
        color=color, orient=ori, linewidth=lw)
    if raw:
        data = (data.sample(n=50, axis=0) if data.shape[0] > 50 else data)
        sns.swarmplot(
            x=cCol, y=yCol, hue=hCol, data=data, ax=ax, color="k", orient=ori)


def barPlot(
        ax, data, yCol, cCol=None, hCol=None, hueDict=None, ci=95, bootN=1000,
        color=None, lw=2, raw=False, ori="v"):
    sns.barplot(
        x=cCol, y=yCol, hue=hCol, data=data, palette=hueDict, ci=ci,
        n_boot=bootN, ax=ax, color=color, orient=ori, linewidth=lw)
    if raw:
        data = (data.sample(n=50, axis=0) if data.shape[0] > 50 else data)
        sns.swarmplot(
            x=cCol, y=yCol, hue=hCol, data=data, ax=ax, color="k", orient=ori)
