#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:02:54 2021

@author: ike
"""


import sys
if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

# from .pipeline.pipelinemain import (
#     processLeicaData, trainMachineLearning)  # , useMachineLearning)
# from .utils.menu import Menu
# from .utils.pathutils import recursive_glob, get_path
# from .aitools.dataset import Dataset
# from .aitools.models.endonet import EndoNet
# from .aitools.trainer import Trainer
#
#
# MAINMENU = Menu("Root Menu -- I want to...")
# MAINMENU["Access leica raw data pipeline"] = [processLeicaData]
# MAINMENU["Train a aitools learning model"] = [trainMachineLearning]
# MAINMENU["Use a trained aitools learning model"] = [useMachineLearning]


# def userInput():
#     saveDirectory = Menu.entryFunctional("save directory", str, True)
#     dataDirectory = Menu.entryFunctional("data directory", str, True)
#     savePath = get_path(saveDirectory, "MODEL TEST", ext="pt")
#     data = recursive_glob(get_path(dataDirectory, "**", "joined-binned*", ext="tif"))
#     dset = Dataset(data, Cout=5, Cin=4, depth=16, Yin=32, Xin=32)
#     model = EndoNet(18, down="max", Din=3, Dout=2, Cin=4, Cout=5)
#     trainer = Trainer(model=model, dataset=dset)
#     trainer.trainingLoop(savePath, 10)
