#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:28:00 2020


@author: ike
"""

import sys
import numpy as np
import pandas as pd
import os.path as op
from time import time as tt
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..main import tqdm
from ..utils.visualization import wait
from ..utils.pathutils import makeParent


class Trainer(object):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __init__(self, model, dataset, weight=True):
        weights = (dataset.weights if weight else ((dataset.weights * 0) + 1))
        weights = torch.tensor(weights).double().to(self.device)

        self.model = model.double().to(self.device)
        self.dataset = dataset
        self.optim = torch.optim.Adam(params=self.model.parameters(), lr=0.001)
        self.decay = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optim, gamma=0.9)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.columns = ["Mode", "Epoch", "Time", "loss", "Total"] + [
            "Class {} Accuracy".format(x) for x in range(self.dataset.Cout)]
        self.dfs = pd.DataFrame(columns=self.columns)

    def _loadCheckpoint(self, checkPath):
        if op.isfile(checkPath):
            checkpoint = torch.load(checkPath, map_location=self.device)
            self.model.load_state_dict(checkpoint["Model"])
            self.dfs = checkpoint["Dataframe"]
            if "Epoch" not in checkpoint:
                return None

            self.optim.load_state_dict(checkpoint["Optimizer"])
            self.decay.load_state_dict(checkpoint["Scheduler"])
            self._makeTrainingPlot()
            return checkpoint["Epoch"] + 1

        makeParent(checkPath)
        return 0

    def _saveCheckpoint(self, checkPath, epoch=None):
        data = dict(Model=self.model.state_dict(), Dataframe=self.dfs)
        if epoch is not None:
            data.update(dict(
                Optimizer=self.optim.state_dict(),
                Scheduler=self.decay.state_dict(),
                Epoch=epoch))
            
        torch.save(data, checkPath)

    def _npTensor(self, tensor):
        return tensor.cpu().detach().numpy()

    def _getStats(self, GT, OUT, mode, epoch, time, loss=np.nan):
        GT, OUT = self._npTensor(GT), np.argmax(self._npTensor(OUT), axis=1)
        OUT = np.ma.masked_where(GT - OUT != 0, OUT)
        stats = np.zeros(len(self.columns), dtype=object)
        stats[:5] = [
            mode, epoch, time, loss, (np.sum(np.equal(GT, OUT)) / GT.size)]
        for cdx in range(self.dataset.Cout):
            pos = np.sum(OUT == cdx)
            tot = np.sum(GT == cdx)
            stats[5 + cdx] = (1 if pos == tot == 0 else (pos / tot))

        self.dfs = pd.concat((self.dfs, pd.DataFrame(
            stats[np.newaxis], columns=self.columns)), axis=0)

    def _trainStep(self, batch, epoch):
        time = tt()
        self.optim.zero_grad()
        In = batch["In"].double().to(self.device)
        GT = batch["GT"].long().to(self.device)
        OUT = self.model(In)  # [N, classes, <D>, H, W]
        loss = self.criterion(OUT, GT)
        loss.backward()
        self.optim.step()
        self._getStats(GT, OUT, "Train", epoch, (tt() - time), loss.item())
        torch.cuda.empty_cache()

    def _checkStep(self, batch, epoch, mode):
        time = tt()
        In = batch["In"].double().to(self.device)
        GT = batch["GT"].long().to(self.device)
        OUT = self.model(In)
        loss = self.criterion(OUT, GT)
        self._getStats(GT, OUT, mode, epoch, (tt() - time), loss.item())
        torch.cuda.empty_cache()

    def _predStep(self, batch):
        In = batch["In"].double().to(self.device)
        OUT = self._npTensor(self.model(In)[0])
        torch.cuda.empty_cache()
        return OUT

    def trainingLoop(self, savePath, epochs=10, batchSize=10):
        holder = self._loadCheckpoint(savePath)
        epoch = (holder if holder is not None else epochs)
        for e in range(epoch, epochs, 1):
            wait("Epoch [{}/{}]".format(e + 1, epochs))

            self.model.train()
            torch.set_grad_enabled(True)
            [self._trainStep(i, e) for i in tqdm(DataLoader(
                self.dataset.train(), batch_size=batchSize, shuffle=True))]

            self.model.eval()
            torch.set_grad_enabled(False)
            [self._checkStep(j, e, "Valid") for j in tqdm(DataLoader(
                self.dataset.valid(), batch_size=batchSize, shuffle=True))]
            self.decay.step()
            self._saveCheckpoint(savePath, e)
            self._makeTrainingPlot()

        if holder is not None:
            self.model.eval()
            torch.set_grad_enabled(False)
            [self._checkStep(k, 0, "Test") for k in tqdm(DataLoader(
                self.dataset.test(), batch_size=batchSize, shuffle=True))]
            self._saveCheckpoint(savePath)

        self._makeTestPlot()
        wait("Training complete")

    def predictionLoop(self, savePath):
        holder = self._loadCheckpoint(savePath)
        if holder is not None:
            return

        self.model.eval()
        torch.set_grad_enabled(False)
        out = np.mean(np.array([self._predStep(p) for p in tqdm(DataLoader(
                self.dataset.predict(), batch_size=1, shuffle=True))]), axis=0)
        out = np.argmax(out, axis=0)
        out[-1, -1] = 4
        plt.imshow(out)
        plt.imsave("/Users/ike/Desktop/Mi4.png", out)
        plt.show()
        plt.close()

    def _makeTrainingPlot(self):
        plt.rcParams["font.size"] = "8"
        sns.set_style("darkgrid")
        clr = sns.color_palette("rocket", n_colors=3)  # "viridis"
        num = len(self.columns) - 5
        fig, ax = plt.subplots(
            nrows=2, ncols=num, figsize=(4 * num, 6), sharey=True, dpi=100)
        X1 = sorted(self.dfs[self.columns[1]].unique().tolist())
        for cdx, col in enumerate(self.columns[5:]):
            data = self.dfs[self.columns[:2] + [col]].copy()
            for mdx, mode in enumerate(data[self.columns[0]].unique()):
                sub = data[data[self.columns[0]] == mode].copy()[
                    [self.columns[1], col]].astype(float)
                X0 = list(range(sub.shape[0]))
                cen = sub.groupby(self.columns[1]).mean()[col].to_numpy()
                spr = sub.groupby(self.columns[1]).sem()[col].to_numpy()
                ax[0, cdx].plot(X0, sub[col], label=mode, lw=1, color=clr[mdx])
                ax[1, cdx].plot(X1, cen, label=mode, lw=1, color=clr[mdx])
                ax[1, cdx].fill_between(
                    X1, cen - spr, cen + spr, color=clr[mdx], alpha=0.5)

            ax[0, cdx].set_title(col)
            ax[0, cdx].set_xlabel("iteration")
            ax[1, cdx].set_xlabel("epoch")
            ax[1, cdx].set_xlim(left=float(X1[0]), right=float(max(X1[-1], 1)))
            ax[0, cdx].set_ylim(bottom=0.0, top=1.0)
            ax[1, cdx].set_ylim(bottom=0.0, top=1.0)
            ax[0, cdx].locator_params(axis="x", nbins=max(len(X1), 2))
            ax[1, cdx].locator_params(axis="x", nbins=max(len(X1), 2))
            if cdx == 0:
                ax[0, cdx].set_ylabel(("accuracy"))
                ax[1, cdx].set_ylabel(("accuracy"))
                ax[0, cdx].locator_params(axis="y", nbins=5)
                ax[1, cdx].locator_params(axis="y", nbins=5)
            if cdx == len(self.columns) - 6:
                ax[0, cdx].legend(loc="upper right", frameon=False)
                ax[1, cdx].legend(loc="upper right", frameon=False)

        plt.tight_layout()
        plt.show()
        plt.close()

    def _makeTestPlot(self):
        plt.rcParams["font.size"] = "8"
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6), dpi=100)
        data = self.dfs[self.dfs[self.columns[0]] == "Test"][self.columns[4:]]
        ax = sns.violinplot(data=data, ax=ax, inner="quartile")
        ax.set_title("Test Accuracies")
        ax.set_ylim(bottom=0.0, top=1.0)
        ax.set_ylabel(("accuracy"))
        ax.locator_params(axis="y", nbins=5)
        plt.tight_layout()
        plt.show()
        plt.close()
