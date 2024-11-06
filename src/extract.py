#!/usr/bin/env python3

# ./extract.py data/data_test.csv

import pandas as pd


def load_data(path):
    """
    load .csv file into panda database
    add some numbered column names
    """
    df = pd.read_csv(path, header=None)

    # check for NaN
    if df.isnull().values.any():
        print("NaN found, exiting program")
        exit()
    return df


def convert(x):
    """
    numerize labels B to 0 and M to 1
    """
    conv = {'B': 0, 'M': 1}
    for i in range(len(x)):
        x.at[i, 1] = conv[x.at[i, 1]]
    return x


def save_csv(arr, name):
    """
    Save a numpy array to a CSV file
    """
    df = pd.DataFrame(arr)
    df.to_csv("data/" + name + ".csv", header=None, index=False)
    print(name + " file saved")


def extract_data(path, val_drop):
    """
    This file is used to extract X and Y arrays from test data
    """
    # load data
    df = load_data(path)

    # extract x and y data
    if isinstance(df.iloc[0, 0], str):
        df = convert(df)
    elif isinstance(df.iloc[0, 1], str):
        df = df.iloc[:, 1:]
        df = df.drop(columns=val_drop)
        df = convert(df)
    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # save data into two x and y .csv files
    save_csv(x, "x_test")
    save_csv(y, "y_test")
