# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Original code by Erin Barnhart, Vicky Mak
# Python port by Ike Ogbonna
# Barnhart Lab, Columbia University 2021
# """
#
#
# import sys
# import numpy as np
# import pandas as pd
# import os.path as op
# import scipy.stats as ss
#
# from ..basepipeline import BasePipeline
#
# from ...main import tqdm
# from ...utils.pipeutils import smooth
# from ...utils.pathutils import getPath
# from ...utils.visualization import wait
# from ...utils.multipagetiff import MultipageTiff
#
#
# class ReceptiveFieldMapper(BasePipeline):
#     threshold = 2
#     stop = False
#
#     def __init__(self):
#         super(ReceptiveFieldMapper, self).__init__()
#
#     def __call__(self):
#         wait(" \nMapping receptive field centers")
#         self.mapReceptiveFieldCenters()
#
#         wait(" \nFiltering mapped responses")
#         self.filterMappedResponses()
#
#         wait(" \nPlotting mapped responses")
#         self.plotMappedResponses()
#
#     @classmethod
#     def mapReceptiveFieldCenters(cls):
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if row("sCheck") or op.isfile(row("rfcFile")) or not row("rCheck"):
#                 continue
#
#             cls.stop = True
#             dfa = pd.read_csv(row("avgFile"))
#             dfa = dfa[dfa[cls.header[3]] == row("channels")[-1]]
#             dfa = dfa.sort_values(
#                 cls.header, ascending=True, ignore_index=True)
#             dfc = dfa[row("strTime")].copy()
#             dfa = dfa.drop(columns=row("strTime"))
#             dfa[cls.rfHead[0]] = dfc.max(axis=1)
#             dfa[cls.rfHead[1]] = dfc.idxmax(axis=1).astype(float)
#             dfa[cls.rfHead[1]] = np.where(
#                 ((dfa[cls.exHead[-1]] % 2) != 0),
#                 ((dfa[cls.rfHead[1]] * 4) - 10),
#                 ((dfa[cls.rfHead[1]] * -4) + 11))
#             dfa[cls.exHead[-1]] = np.where(
#                 (dfa[cls.exHead[-1]] < 3), 0, 1)
#             dfc = dfa[cls.header].copy().drop_duplicates(
#                 keep="first", ignore_index=True)
#             for idx, item in enumerate(cls.rfHead[2:4]):
#                 dfc[item[0]] = dfa[dfa[cls.exHead[-1]] == idx].groupby(
#                     cls.header).mean().reset_index()[cls.rfHead[1]]
#                 dfc[item[1]] = dfa[dfa[cls.exHead[-1]] == idx].groupby(
#                     cls.header).agg(np.std, ddof=0).reset_index()[
#                     cls.rfHead[1]]
#
#             dfa = dfa.filter(cls.header + [cls.rfHead[0]])
#             dfc[cls.rfHead[4][0]] = dfa.groupby(
#                 cls.header).mean().reset_index()[cls.rfHead[0]]
#             dfc[cls.rfHead[4][1]] = dfa.groupby(cls.header).agg(
#                 np.std, ddof=0).reset_index()[cls.rfHead[0]]
#             stds = [item[1] for item in cls.rfHead[2:4]]
#             dfc[cls.rfHead[-1]] = np.where(
#                 dfc[stds].max(axis=1) < cls.threshold, 1, 0)
#             dfc.to_csv(row("rfcFile"), encoding="utf-8", index=False)
#
#         if cls.stop:
#             sys.exit(
#                 ("Stop and conduct quality control on all of your RF"
#                  " centers.csv files"))
#
#     @classmethod
#     def filterMappedResponses(cls):
#         df_ = dict()
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if not op.isfile(row("rfcFile")) or not row("sCheck"):
#                 continue
#
#             dfc = pd.read_csv(
#                 row("rfcFile"), usecols=(cls.header[:3] + [cls.rfHead[-1]]))
#             dfc = dfc[dfc[cls.rfHead[-1]] > 0]
#             if dfc.empty:
#                 continue
#
#             for old, new in (("mesFile", "nmsFile"), ("avgFile", "mapFile")):
#                 if not op.isfile(row(old)) or op.isfile(row(new)):
#                     continue
#
#                 dfn = pd.merge(pd.read_csv(
#                     row(old)), dfc, how="inner", on=cls.header[:3])
#                 if not dfn.empty:
#                     df_[row(new)] = (
#                         pd.concat(
#                             (df_[row(new)], dfn), axis=0, ignore_index=True)
#                         if row(new) in df_ else dfn)
#
#         for new, dfn in df_.items():
#             if not op.isfile(new):
#                 dfn.to_csv(new, encoding="utf-8", index=False)
#
#     @classmethod
#     def plotMappedResponses(cls):
#         def makePlot(row, dfs, name):
#             fdx = dfs[cls.header[0]].unique().tolist()
#             fdx = len(set(["-".join(f.split("-")[:4]) for f in fdx]))
#             rdx = dfs.shape[0]
#             adx = dfs[cls.imDims[3]].sum()
#             name = "{} Fly N: {}, ROI N: {}, Response N: {}".format(
#                 name, fdx, rdx, adx)
#             cen, spr = dict(), dict()
#             for c in row("channels"):
#                 dfc = dfs[dfs[cls.header[3]] == c].copy()
#                 dfc = np.squeeze(dfc.filter(items=row("strTime")).to_numpy())
#                 cen[c] = smooth(dfc if dfc.ndim == 1 else np.mean(dfc, axis=0))
#                 spr[c] = (ss.sem(dfc, axis=0) if dfc.ndim > 1 else None)
#
#             MultipageTiff.addFigure(
#                 row("timing"), cen, title=name, axes=("Time (s)", "DF/F"),
#                 dY=spr, light=row("lightOn"))
#
#         for row in tqdm(cls.iterRows(), total=cls.len()):
#             if not op.isfile(row("mapFile")) or not(row("sCheck")):
#                 continue
#
#             dfm = pd.read_csv(row("mapFile"))
#             fig = row("maprFig")
#             for lay in dfm[cls.header[1]].unique().tolist():
#                 save = getPath("-".join((fig, lay)), ext="tif")
#                 if op.isfile(save):
#                     continue
#
#                 dfs = dfm[dfm[cls.header[1]] == lay].copy()
#                 makePlot(row, dfs, "{} Total".format(lay))
#                 for name in dfs[cls.header[0]].unique().tolist():
#                     dff = dfs[dfs[cls.header[0]] == name].copy()
#                     makePlot(row, dff, name)
#
#                 MultipageTiff.saveFigures(save)
