#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


STIM = dict(
    frm="frames",
    vol="AIN4",
    gbt="global_time",
    rlt="rel_time",
    smt="stim_type",
    enm="epoch_number",
    epc="epoch")


RESP = dict(
    reg="region",
    roi="ROI",
    szs="sizes",
    chn="channel",
    avg="mean_PI",
    rnm="response_number",
    nrs="response N",
    enm=STIM["enm"],
    rmx="rmax",
    pmx="pmax",
    xst=("x", "x_std"),
    yst=("y", "y_STD"),
    ast=("amplitude", "amplitude_std"),
    map="mappable?",
    int="integrated response",
    prv="pearsons R value",
    ppv="p-value")


MAIN = dict(
    dat="Date",
    fly="Fly",
    cel="Cell",
    lay="Layers",
    zpl="Z Plane",
    lif=".lif Name",
    lnm=".lif Num",
    stn="Stim Name",
    stt="Stim Timestamp",
    ch0="Channel 0",
    ch1="Channel 1",
    bl0="Begin Baseline",
    bl1="End Baseline",
    ref="Reference Channel")
