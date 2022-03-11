# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Original code by Erin Barnhart, Vicky Mak
# Python port by Ike Ogbonna
# Barnhart Lab, Columbia University 2021
# """
#
#
# import numpy as np
# import pandas as pd
# import os.path as op
#
# from ..basepipeline import BasePipeline
#
# from ...main import tqdm
# from ...utils.pathutils import getPath, getName, changeExt
# from ...utils.pipeutils import boundInt, smooth
# from ...utils.visualization import wait
# from ...utils.multipagetiff import MultipageTiff
#
#
# class ResponseMeasurer(BasePipeline):
#     cStr = "mean_PI"
#
#     def __init__(self):
#         super(ResponseMeasurer, self).__init__()
#
#     def __call__(self):
#         wait(" \nMeasuring and plotting raw responses")
#         self.measureAndPlotRawResponses()
#
#         wait(" \nMeasuring individual responses")
#         self.measureIndividualResponses()
#
#         wait(" \nMeasuring and plotting average responses")
#         self.measureAndPlotAverageResponses()
#
#         wait(" \nMeasuring and plotting responses from binned data")
#         self.measureAndPlotBinnedResponses()
#
#     @classmethod
#     def measureAndPlotRawResponses(cls):
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if op.isfile(row("rawFile")) and op.isfile(row("mesFile")):
#                 continue
#
#             dfr = None
#             bkg = dict()
#             col = [str(x + 1) for x in range(row("frames"))]
#             for c in row("channels"):
#                 # load image stack and apply background correction
#                 mpt = MultipageTiff(getPath(row("cPath", c), ext="tif"))
#                 bkg[c] = mpt.correct(row("bkgTif"))
#                 for mskFile in row("mskDirs"):
#                     res, ROI, sizes = mpt.getMaskedROIs(mskFile)
#                     dfs = pd.DataFrame(data=res, columns=col)
#                     dfs.insert(0, cls.cStr, np.mean(res, axis=-1))
#                     dfs.insert(0, cls.header[0], getName(row("sample")))
#                     lay = getName(row("sample")).split("-") + [
#                         changeExt(getName(mskFile))]
#                     dfs.insert(1, cls.header[1], "-".join(lay[5:]))
#                     dfs.insert(2, cls.header[2], ROI)
#                     dfs.insert(3, cls.header[3], c)
#                     dfs.insert(4, cls.exHead[0], sizes)
#                     dfr = (dfs if dfr is None else pd.concat(
#                         (dfr, dfs), axis=0, ignore_index=True))
#
#             dfr.to_csv(row("rawFile"), encoding="utf-8", index=False)
#             dfr[cls.header + [cls.exHead[0]]].to_csv(
#                 row("mesFile"), encoding="utf-8", index=False)
#             dfr[cls.header[1]] = (
#                 dfr[cls.header[1]].astype(str) + " ROI " +
#                 dfr[cls.header[2]].astype(str))
#             icl = np.array(range(row("frames"))) + 1
#             for name in dfr[cls.header[1]].unique().tolist():
#                 raw = dfr[dfr[cls.header[1]] == name].copy()
#                 raw = {
#                     c: smooth(raw[raw[cls.header[3]] == c][col].to_numpy()[0])
#                     for c in row("channels")}
#                 MultipageTiff.addFigure(
#                     icl, raw, title=name, axes=("Frame", "F raw"), bkg=bkg)
#
#             MultipageTiff.saveFigures(row("rawrFig"))
#
#     @classmethod
#     def measureIndividualResponses(cls):
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if op.isfile(row("indFile")):
#                 continue
#
#             # calc median global/relative stimulus time per frame
#             dff = pd.read_csv(row("frmStim"), usecols=(
#                 cls.rrHead[:-1] if row("sCheck") else cls.rrHead))
#             dff = dff.drop_duplicates(
#                 subset=cls.rrHead[2], keep="first", ignore_index=True)
#             dff = dff.sort_values(cls.rrHead[-2], ascending=True)
#             rel_time = dff[cls.rrHead[1]].to_numpy()
#
#             # epoch first index, first frame of each epoch
#             try:
#                 efi = (rel_time[:-1] > (rel_time[1:] + row("offset")))
#                 efi = list(set([0] + (np.nonzero(efi)[0] + 1).tolist()))
#                 eln = (None if row("sCheck") else dff[
#                     cls.rrHead[-1]].to_numpy()[efi])
#                 efi[0] = efi[1] - row("eFrames")
#             except IndexError:
#                 continue
#
#             # Find, resample, normalize, save responses to single epochs
#             dfr = pd.read_csv(row("rawFile"))
#             dfi = None
#             for idx, edx in enumerate(efi):
#                 dft = dfr[cls.header].copy()
#                 dft[cls.exHead[1]] = idx + 1
#                 if not row("sCheck"):
#                     dft[cls.exHead[-1]] = eln[idx]
#
#                 col = [
#                     str(boundInt((edx + x), 1, row("frames")))
#                     for x in range(row("eFrames"))]
#                 res = dfr[col].to_numpy()
#                 bln = np.median(
#                     res[:, :row("frameOn")], axis=-1)[:, np.newaxis]
#                 res = (res - bln) / bln
#                 dft = pd.concat((dft, pd.DataFrame(
#                     res, columns=row("strTime"))), axis=1)
#                 dfi = (dft if dfi is None else pd.concat(
#                     (dfi, dft), axis=0, ignore_index=True))
#
#             col = cls.header + (
#                 [cls.exHead[1]] if row("sCheck") else cls.exHead[1:])
#             dfi = dfi.sort_values(col, ascending=True, ignore_index=True)
#             dfi.to_csv(row("indFile"), encoding="utf-8", index=False)
#
#     @classmethod
#     def measureAndPlotAverageResponses(cls):
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if op.isfile(row("avgFile")) or not op.isfile(row("indFile")):
#                 continue
#
#             col = cls.header + ([] if row("sCheck") else [cls.exHead[2]])
#             dfi = pd.read_csv(row("indFile"))
#             dfa = dfi.drop(columns=[cls.exHead[1]], axis=1).groupby(
#                 col).mean().reset_index()
#             dfa[cls.imDims[3]] = dfi[cls.exHead[1]].max()
#             dfa.to_csv(row("avgFile"), encoding="utf-8", index=False)
#
#             # generate and save plots of average per-epoch responses
#             dfa[cls.header[1]] = (
#                 dfa[cls.header[1]].astype(str) + " ROI " +
#                 dfa[cls.header[2]].astype(str) + " Response N: " +
#                 dfa[cls.imDims[3]].astype(str))
#             for name in dfa[cls.header[1]].unique().tolist():
#                 dfr = dfa[dfa[cls.header[1]] == name].copy().sort_values(
#                     cls.header, ascending=True, ignore_index=True)
#                 if row("sCheck"):
#                     avg = {c: smooth(dfr[dfr[cls.header[3]] == c][row(
#                         "strTime")].to_numpy()[0]) for c in row("channels")}
#                 else:
#                     dfr[cls.header[3]] = dfr[cls.header[3]].astype(str) + (
#                         dfr[cls.exHead[-1]].astype(str))
#                     avg = {c: smooth(dfr[dfr[cls.header[3]] == c][row(
#                         "strTime")].to_numpy()[0]) for c in
#                            dfr[cls.header[3]].unique().tolist()}
#
#                 MultipageTiff.addFigure(
#                     row("timing"), avg, title=name,
#                     axes=("Time (s)", "Average DF/F"))
#
#             MultipageTiff.saveFigures(row("measFig"))
#
#     @classmethod
#     def measureAndPlotBinnedResponses(cls):
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if op.isfile(row("bnaFile")):
#                 continue
#
#             dfr = None
#             bkg = dict()
#             for c in row("channels"):
#                 # load image stack and apply background correction
#                 mpt = MultipageTiff(row("binTif", c))
#                 bkg[c] = mpt.correct(row("bkgTif"))
#                 col = [str(x + 1) for x in range(len(mpt))]
#                 for mskFile in row("mskDirs"):
#                     res, ROI, sizes = mpt.getMaskedROIs(mskFile)
#                     dfs = pd.DataFrame(data=res, columns=col)
#                     dfs.insert(0, cls.cStr, np.mean(res, axis=-1))
#                     dfs.insert(0, cls.header[0], getName(row("sample")))
#                     lay = getName(row("sample")).split("-") + [
#                         changeExt(getName(mskFile))]
#                     dfs.insert(1, cls.header[1], "-".join(lay[5:]))
#                     dfs.insert(2, cls.header[2], ROI)
#                     dfs.insert(3, cls.header[3], c)
#                     dfs.insert(4, cls.exHead[0], sizes)
#                     dfr = (dfs if dfr is None else pd.concat(
#                         (dfr, dfs), axis=0, ignore_index=True))
#
#             dfr.to_csv(row("bnaFile"), encoding="utf-8", index=False)
#             dfr[cls.header[1]] = (
#                 dfr[cls.header[1]].astype(str) + " ROI " +
#                 dfr[cls.header[2]].astype(str))
#             icl = [int(x) for x in col]
#             for name in dfr[cls.header[1]].unique().tolist():
#                 raw = dfr[dfr[cls.header[1]] == name].copy()
#                 raw = {
#                     c: smooth(raw[raw[cls.header[3]] == c][col].to_numpy()[0])
#                     for c in row("channels")}
#                 MultipageTiff.addFigure(
#                     icl, raw, title=name, axes=("Frame", "F binned"), bkg=bkg)
#
#             MultipageTiff.saveFigures(row("bnavFig"))
