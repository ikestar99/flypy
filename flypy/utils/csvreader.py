#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


import numpy as np
import pandas as pd


class CSVReader(object):
    @classmethod
    def fromFile(cls, file):
        """
        Instantiaate CSVReader object

        @param file: complete file path to imaging settings.csv file
        @type file: string
        """
        # load entire csv file as 2D pandas dataframe
        reader = CSVReader()
        reader.dfs = pd.read_csv(file)
        return reader

    @classmethod
    def fromDataFrame(cls, dfs):
        """
        Instantiaate CSVReader object

        @param dfs: complete file path to imaging settings.csv file
        @type dfs: string
        """
        # load entire csv file as 2D pandas dataframe
        reader = CSVReader()
        reader.dfs = dfs
        return reader

    def __init__(self):
        self.dfs = None
        self.index = 0

    def __len__(self):
        return self.dfs.shape[0]

    def __contains__(self, key):
        return key in self.dfs.columns

    def __getitem__(self, idx):
        item = self.dfs.iloc[idx].to_dict()
        item = {k: (None if self.isNan(i) else i) for k, i in item.items()}
        return item

    def __setitem__(self, idx, value):
        for key, item in value.items():
            self.dfs.loc[idx, key] = item

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            self.index = 0
            raise StopIteration
        else:
            row = self[self.index]
            self.index += 1
            return row

    @staticmethod
    def isNan(item):
        value = (True if item is None else False)
        value = (
            True if (type(item) == np.float64 and np.isnan(item)) else value)
        value = (True if str(item) == "nan" else value)
        return value

    def empty(self):
        self.dfs = pd.DataFrame(columns=self.dfs.columns)

    def dropna(self, *args):
        args = [arg for arg in args if arg in self]
        self.dfs = self.dfs.dropna(axis=0, subset=args)

    def save(self, file):
        self.dfs.to_csv(file, encoding="utf-8", index=False)

    def getColumn(self, key, unique=False):
        columns = (
            self.dfs[key].unique() if unique else self.dfs[key].to_numpy())
        return columns.flatten()

    def filterRows(self, item, unique=False):
        """
        Extract all imaging CSV entries that match a patern defined by item

        @param item: key: value pairings where keys correspond to column
            headers in imaging.csv file and values are those desired for a
            particular use case
        @type item: dict

        @return: list of dictionaries where each dictionary corresponds to a
            row in the imaging.csv file that adheres to the item template
        @rtype: list
        """
        dfs = self.dfs.copy()
        for key, item in item.items():
            dfs = dfs[dfs[key] == item]

        dfs = (dfs.drop_duplicates() if unique else dfs)
        return CSVReader.fromDataFrame(dfs)

    def filterColumns(self, *args, unique=False):
        """
        Get all values in column(s) of CSV

        @param args: column headers from which to extract unique values
        @type args: arguments

        @return: list of dictionaries with all unique {arg: value} combinations
        @rtype: list
        """
        dfs = self.dfs[list(args)].copy()
        dfs = (dfs.drop_duplicates() if unique else dfs)
        return CSVReader.fromDataFrame(dfs)
