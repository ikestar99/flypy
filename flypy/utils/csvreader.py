#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:41:02 2021

@author: ike
"""


import pandas as pd


"""
Helper functions for loading and modifying data in .csv files using pandas
backend.
"""


def load_csv(
        file: str
):
    """
    Load a CSV file into a DataFrame.

    Args:
        file (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: DataFrame containing the loaded CSV data.

    Examples:
        >>> test_data = load_csv("data.csv")
        >>> print(test_data.head())
           Name  Age  Salary
        0  John   30   50000
        1  Emma   25   45000
        2  Mark   35   60000
    """
    data = pd.read_csv(file)
    return data


def save_csv(
        file: str,
        dataframe: pd.DataFrame
):
    """
    Save a DataFrame as a CSV file.

    Args:
        file (str): Path to save the CSV file.
        dataframe (pandas.DataFrame): DataFrame to be saved.

    Examples:
        >>> test_data = pd.DataFrame({
        ...    "Name": ["John", "Emma", "Mark"],
        ...    "Age": [30, 25, 35],
        ...    "Salary": [50000, 45000, 60000]})
        >>> test_file = "path/to/file.csv"
        >>> save_csv(test_file, test_data)
    """
    dataframe.to_csv(file, encoding="utf-8", index=False)


def save_data_as_csv(
        data,
        file: str,
        columns: list = None
):
    """
    Save data as a CSV file.

    Args:
        data (list, dict, numpy.ndarray, pandas.DataFrane): Data to be saved.
        columns (list): Ordered column names if not already present in data.
        file (str): The path to save the CSV file.

    Examples:
        >>> test_data = [
        ...     ["John", 30, 50000],
        ...     ["Emma", 25, 45000],
        ...     ["Mark", 35, 60000]]
        >>> test_columns = ["Name", "Age", "Salary"]
        >>> test_file = "path/to/file.csv"
        >>> save_data_as_csv(test_data, test_file, test_columns)
    """
    dataframe = pd.DataFrame(data, columns=columns)
    save_csv(file, dataframe)


def empty(
        dataframe: pd.DataFrame
):
    """
    Create an empty DataFrame with identical column structure .

    Args:
        dataframe (pandas.DataFrame): DataFrame to use as a template.

    Returns:
        pandas.DataFrame: Empty DataFrame with identical columns to the input.

    Examples:
        >>> test_data = pd.DataFrame({
        ...     "Name": ["John", "Emma", "Mark"],
        ...     "Age": [30, 25, 35],
        ...     "Salary": [50000, 45000, 60000]})
        >>> empty_data = empty(test_data)
        >>> print(empty_data)
        Empty DataFrame
        Columns: [Name, Age, Salary]
        Index: []
    """
    empty_dataframe = pd.DataFrame(columns=dataframe.columns)
    return empty_dataframe


def pattern_match(
        dataframe: pd.DataFrame,
        pattern: dict
):
    """
    Filter rows of a DataFrame based on a pattern match.

    Args:
        dataframe (pandas.DataFrame): DataFrame to be filtered.
        pattern (dict): Key-value pairs representing the match criteria. Each
            key in pattern corresponds to a column in dataframe. The colums are
            filtered such that every column only contains the value specified
            by pattern[key]. Column not referenced in pattern are left in
            place.

    Returns:
        pandas.DataFrame: Filtered DataFrame.

    Examples:
        # Filter one column
        >>> test_data = pd.DataFrame({
        ...     "Name": ["Erin", "Zack", "Mark", "John"],
        ...     "Age": [30, 25, 35, 30],
        ...     "Salary": [50000, 55000, 60000, 65000]})
        >>> test_pattern = {"Age": 30}
        >>> filtered_data = pattern_match(test_data, test_pattern)
        >>> print(filtered_data)
           Name  Age  Salary
        0  Erin   30   50000
        3  John   30   65000

        # Filter multiple columns
        >>> test_pattern = {"Name": "John", "Age": 30}
        >>> filtered_data = pattern_match(test_data, test_pattern)
        >>> print(filtered_data)
           Name  Age  Salary
        3  John   30   65000
    """
    for key, item in pattern.items():
        dataframe = dataframe[dataframe[key] == item]

    return dataframe
