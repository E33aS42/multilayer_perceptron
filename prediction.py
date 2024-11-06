#!/usr/bin/env python3

# usage: ./prediction.py nn_model.pkl

import numpy as np
import pandas as pd
from src.helperfile import helper_prediction
from src.network import network
import pickle
import joblib
import sys
import os
from src.extract import extract_data


def load_model(filepath):
    with open(filepath, "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model


def load_data(path):
    """
    load .csv file into numpy array
    """
    return np.loadtxt(path, delimiter=",")


def save_csv(arr, name):
    """
    Save a numpy array to a CSV file
    """
    df = pd.DataFrame(arr)
    df.to_csv("data/" + name + ".csv", header=None, index=False)
    print(name + " file saved")


if __name__ == "__main__":
    try:
        cnt = len(sys.argv)
        assert cnt > 1, "missing arguments\n" + \
            helper_prediction.__doc__

        # load model
        model = sys.argv[1]
        ext = os.path.splitext(model)[-1].lower()
        assert ext == ".pkl", "incorrect model file path"
        [hidden, lr, epochs, prms, init, hidden_a, output_a, optim, gradients,
            weighted_gradients, moving_average] = load_model('nn_model.pkl')

        # check for column drop
        val_drop = []
        if cnt > 2:
            j = 3
            while (j < cnt):
                assert sys.argv[j].isdigit(), "column drop argument must be a positive int"
                val_drop.append(int(sys.argv[j]))
                j += 1

        # load test data
        x_test = load_data("data/x_test.csv")
        y_test = load_data("data/y_test.csv")

        # load model parameters
        model = network(hidden_layers=hidden,
                        lr=lr,
                        n_epochs=epochs,
                        hidden_a=hidden_a,
                        output_a=output_a,
                        optim=optim,
                        prms=prms,
                        gradients=gradients,
                        weighted_gradients=weighted_gradients,
                        moving_average=moving_average)

        # apply normalization
        my_Scaler = joblib.load('scaler.pkl')
        X_test = my_Scaler.transform(x_test)

        # transpose matrices
        X_test = X_test.T
        save_csv(X_test, "X_test")
        Y_test = y_test.reshape((1, y_test.shape[0]))
        print('dimensions of X_test:', X_test.shape)
        print('dimensions of y_test:', Y_test.shape)

        # make prediction
        model.prediction(X_test, Y_test)

    except Exception as e:
        print(e)
        exit()
