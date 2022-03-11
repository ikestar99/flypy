#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 03:04:49 2021

@author: ike
"""


import numpy as np

import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def makePrediction(model, batch, pbar=None):
    out = model(batch["In"].double().to(DEVICE)).cpu().detach().numpy()
    if pbar is not None:
        pbar.update(1)

    # fix this -- should not be averaging across batches
    return np.mean(out, axis=0)


def getAveragePrediction(model, loader, pbar):
    model.eval(); torch.set_grad_enabled(False)
    runAvg = np.mean(np.array(
        [makePrediction(model, batch, pbar) for batch in loader]), axis=0)
    return runAvg


# def maskAndCorrelateFolders(cls):
#     def watershedMask(waterMap, maskSave, name):
#         newMask = scn.binary_erosion(sks.watershed(
#             -waterMap, mask=(waterMap != 0), watershed_line=True))
#         MultipageTiff.saveImage(
#             newMask, getPath(maskSave, name, ext="tif"))
#
#     def getPixelSeries(row, col):
#         return cls.mtpTiff[..., row // cls.kSize, col // cls.kSize]
#
#     def assignLabel(mask, corrRefs, seed, Y, X):
#         for row in (Y - 1, Y, Y + 1):
#             for col in (X - 1, X, X + 1):
#                 if all(((row < mask.shape[0]), (col < mask.shape[1]),
#                         (mask[row, col] == 1))):
#                     if all(((0 < row < mask.shape[0] - 1),
#                             (0 < col < mask.shape[1] - 1))):
#                         tile = mask[row - 1:row + 2, col - 1:col + 2]
#                         if np.unique((tile * (tile > 1))).size > 2:
#                             continue
#
#                     corr = [scs.pearsonr(corrRefs[x], getPixelSeries(
#                         row, col))[0] for x in range(len(corrRefs))]
#                     if (corr[seed - 2] * cls.margin) >= max(corr):
#                         mask[row, col] = seed
#                         corrRefs[seed - 2] += getPixelSeries(row, col)
#                         mask, corrRefs = assignLabel(
#                             mask, corrRefs, seed, row, col)
#
#         return mask, corrRefs
#
#     # flyCFGs = getFlyCFGs()
#     # with tqdm(total=0) as pbar:
#     #     for flyCFG in flyCFGs.values():
#     #         model, _ = getModel(flyCFG)
#     #         for sample, ids in cls.getUnique("idList"):
#     #             row = cls.getRow(sample, ids[0])
#     #             save = getPath(row("predDir"), flyCFG["Name"], ext="npy")
#     #             if op.isfile(save):
#     #                 continue
#     #
#     #             flyCFG["DataDir"] = sample
#     #             flyCFG.Data["Channel"] = getName(
#     #                 changeExt(row("binTif", row("channels")[-1])))
#     #             flyCFG.resize("Hin", row("shape")[0])
#     #             flyCFG.resize("Win", row("shape")[1])
#     #             loader = toPipeline(flyCFG)
#     #             pbar.total += len(loader)
#     #             pbar.set_description(flyCFG["Name"], refresh=True)
#     #             avgPred = getAveragePrediction(
#     #                 model, loader, pbar=pbar)[
#     #                       :, :row("shape")[0], :row("shape")[1]]
#     #             np.save(save, avgPred)
#
#     # pbar.close()
#     # del pbar
#     # for row in tqdm(cls.iterRows(), total=cls.len()):
#     #     if op.isfile(row("bkgTif")):
#     #         continue
#     #
#     #     sys.exit(("Masking with machine learning is still under "
#     #               "development. Please make 'background.tif' and "
#     #               "ROI mask files for each of your samples and return"))
#         # BNet1 = np.load(getPath(
#         #     row("predDir"), flyCFGs["B1"]["Name"], ext="npy"))
#         # CNet0 = np.load(getPath(
#         #     row("predDir"), flyCFGs["C0"]["Name"], ext="npy"))
#         # mask = (np.argmax(BNet1, axis=0) == 0).astype(int)
#         # MultipageTiff.saveImage(mask, row("bkgTif"), normalize=True)
#         # mask = (mask == 0) * CNet0[cls.ccdx]
#         #
#         # # PREDICTION WATERSHED
#         # watershedMask(mask, row("maskDir"), "probability watershed mask")
#         #
#         # # CORRELATIONAL WATERSHED
#         # keys = [key for key in ids if "moving" in key]
#         # if len(keys) == 0:
#         #     continue
#         #
#         # id = keys[0]
#         # channels = self.varValue(sample, id, mode="C")
#         # channel = (channels["ch01"] if "ch01" in channels else
#         #            channels["ch00"])
#         # self.mtpTiff = MultipageTiff(getPath(
#         #     sample, id, "".join((self.bindDir, channel)),
#         #     ext="tif"))
#         # self.mtpTiff.blockReduce(self.kSize)
#         #
#         # corrMask = np.zeros(mask.shape)
#         # corrMask[tuple(skf.peak_local_max(mask).T)] = 1
#         # corrMask, features = scn.label(
#         #     corrMask, structure=np.ones((3, 3)))
#         # mask = corrMask + (mask > 0).astype(int)
#         # corrRefs = []
#         # for feature in range(features):
#         #     YX = np.argwhere(mask == feature + 2)
#         #     corrRefs += [np.sum(np.array(
#         #         [getPixelSeries(YX[j, 0], YX[j, 1])
#         #          for j in range(YX.shape[0])]), axis=0)]
#         #
#         # for feature in range(features):
#         #     YX = np.argwhere(mask == feature + 2)
#         #     for j in range(YX.shape[0]):
#         #         mask, corrRefs = assignLabel(
#         #             mask=mask, corrRefs=corrRefs, seed=(feature + 2),
#         #             Y=YX[j, 0], X=YX[j, 1])
#         #
#         # MultipageTiff.saveImage(
#         #     mask, getPath(
#         #         row("maskDir"), "correlation watershed mask", ext="tif"))
#
#     # sys.exit(("Please take a moment to review the masks generated by "
#     #           "machine learning and update them as necessary"))
