#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


import pandas as pd


class CSVReader(object):
    def __init__(self, file):
        """
        Instantiaate CSVReader object

        @param file: complete file path to imaging settings.csv file
        @type file: string
        """
        # load entire csv file as 2D pandas dataframe
        self.dfs = pd.read_csv(file)
        self.index = 0

    def __len__(self):
        return self.dfs.shape[0]

    def __getitem__(self, item):
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

        dfs = dfs.to_dict(orient="records")
        return dfs

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self):
            self.index = 0
            raise StopIteration
        else:
            row = self.dfs.iloc[self.index].to_dict()
            self.index += 1
            return row

    def __call__(self, *args):
        """
        Get all unique values in column of CSV

        @param args: column headers from which to extract unique values
        @type args: arguments

        @return: list of dictionaries with all unique {arg: value} combinations
        @rtype: list
        """
        items = self.dfs[list(args)].copy().drop_duplicates().to_dict(
            orient="records")
        return items

    def getColumnSet(self, key):
        items = self.dfs[key].copy().drop_duplicates().to_list()
        return items
