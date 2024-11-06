#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import seaborn as sns

'''
Part 1:

Split data into training and validation datasets.

'''


def convert(x):
    """
    numerize labels B to 0 and M to 1
    """
    conv = {'B': 0, 'M': 1}
    for i in range(len(x)):
        x.at[i, 1] = conv[x.at[i, 1]]
    return x


def load_data(path):
    """
    load .csv file into panda database
    add some numbered column names
    numerize labels B to 0 and M to 1
    """
    df = pd.read_csv(path, header=None)
    if df.iloc[1, 1] == 'M' or df.iloc[1, 1] == 'B':
        x = df.iloc[:, 1:]
        print(x.head())
        x = convert(x)
    else:
        x = df
        print(x.head())

    return x


def save_csv(arr, name):
    """
    Save the DataFrame to a CSV file
    """
    df = pd.DataFrame(arr)
    df.to_csv("data/" + name + ".csv", header=None, index=False)
    print(name + " file saved")


def split_data(x, y, p_train, p_val):
    """
    shuffle data while applying a seed
    return x and y splits into training and test or validation sets
    given a split percentage
    """
    # shuffle input data
    x_shuffle = x[:]
    y_shuffle = y[:]
    np.random.seed(4242)
    np.random.shuffle(x_shuffle, )
    np.random.seed(4242)
    np.random.shuffle(y_shuffle, )

    # extract splitted data
    m = x.shape[0]
    idx_tr = int(m * p_train)
    idx_val = int(m * (p_train + p_val))
    x_train = x_shuffle[:idx_tr]
    x_val = x_shuffle[idx_tr:idx_val]
    x_test = x_shuffle[idx_val:]
    y_train = y_shuffle[:idx_tr]
    y_val = y_shuffle[idx_tr:idx_val]
    y_test = y_shuffle[idx_val:]

    return (x_train, x_val, x_test, y_train, y_val, y_test)


if __name__ == "__main__":

    try:
        # check input
        cnt = len(sys.argv)
        assert cnt > 1, "missing argument"
        path = sys.argv[1]
        ext = os.path.splitext(path)[-1].lower()
        assert ext == ".csv", "incorrect file type. It should be an .csv file"
        val_drop = []
        if cnt > 2:
            j = 2
            while (j < cnt):
                assert sys.argv[j].isdigit(), "column drop argument must be a positive int"
                val_drop.append(int(sys.argv[j]))
                j += 1
        df = load_data(path)
        corr = df.corr()
        sns.heatmap(corr, cmap="Blues", annot=True)
        plt.title("Original correlation heatmap")

        column_names = list(df.columns.values)
        plt.figure()
        df1 = df.drop(columns=val_drop)
        corr = df1.corr()
        sns.heatmap(corr, cmap="Blues", annot=True)
        plt.title("Correlation heatmap after removing biggest correlated variables")

        data = df1.values
        save_csv(data, "data_nocorr")

        # extract x and y data
        x = df1.iloc[:, 1:].values
        y = df1.iloc[:, 0].values

        # split data
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(x, y, 0.65, 0.15)
        print("train shapes:", x_train.shape, y_train.shape)
        print("valid shapes:", x_val.shape, y_val.shape)

        # save split into .csv files
        save_csv(x_train, "x_train")
        save_csv(x_val, "x_val")
        save_csv(x_test, "x_test")
        save_csv(y_train, "y_train")
        save_csv(y_val, "y_val")
        save_csv(y_test, "y_test")

        plt.show()

    except Exception as e:
        print(e)
