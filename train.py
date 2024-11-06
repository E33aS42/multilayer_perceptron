#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.scaler import Minmax_Scaler, Standard_Scaler
from src.network import network
from src.helperfile import helper_train
from src.parsing import parser
import sys
import joblib

'''
Part 3:

Train model on training and validation datasets.
'''


def load_data(path):
    """
    load .csv file into numpy array
    """
    return np.loadtxt(path, delimiter=",")


def save_csv(arr, name):
    df = pd.DataFrame(arr)
    # Save the DataFrame to a CSV file
    df.to_csv("data/" + name + ".csv", header=None, index=False)
    print(name + " file saved")


if __name__ == "__main__":

    # try:
    # Default parameters
    hidden_layers = (32, 16)
    lr = 0.3
    n_epochs = 1000
    init = "random"
    hidden_a = "sigmoid"
    output_a = "softmax"
    optim = "none"
    early = "no"
    patience = 10
    multi = "no"
    display = "no"

    # If multiple trainings:
    cases = [((32, 16), (16, 8), (32, 16, 8)), (0.1, 0.4), ('sigmoid',), ('softmax1',), ('none',)]
    # cases = [((16, 8),), (0.2,), ('relu',), ('softmax',), (('none',), ('mom',))]
    cases = [((16, 8),), (0.01,), ('sigmoid',), ('sigmoid',), (('none',), ('mom', 0.9, 0.999), ('rms', 0.9, 0.999), ('adam', 0.9, 0.999))]
    # cases = "[((16, 8),), (0.02,), ('sigmoid',), ('softmax1',), ('none', ('adam',0.9, 0.999), ('adam', 0.1,0.999))]"
    # cases = [((16, 8),), (0.1,), ('sigmoid', 'tanh', 'relu', 'leaky'), ('sigmoid',), (('none',),)]
    # cases = [((32, 16),), (0.3,), ('sigmoid',), ('sigmoid', 'softmax','softmax1'), (('none',),)]

    argv = sys.argv
    parse = parser()

    length = len(argv)

    # get helperfile if requested
    if length > 1 and argv[1] == "-h":
        print(helper_train.__doc__)
        exit()

    # Get hidden layers input if any
    if "-hidden" in argv:
        index = argv.index("-hidden")
        assert index < length - 2, "network should have at least 2 hidden layers"
        hidden_layers = parse.get_hidden_arg(argv, index)

    # Get any modulable inputs if any
    list_modules = ["-epochs", "-lr", "-init", "-hidacti",
                    "-outacti", "-opti", "-early", "-multi", "-display"]

    for mod in list_modules:
        if mod in argv:
            index = argv.index(mod)
            assert index < length - 1, "missing argument"
            if mod == "-epochs":
                n_epochs = parse.get_epochs_arg(argv, index)
            if mod == "-lr":
                lr = parse.get_lr_arg(argv, index)
            if mod == "-init":
                init = parse.get_init_arg(argv, index)
            if mod == "-hidacti":
                hidden_a = parse.get_hidden_acti_arg(argv, index)
            if mod == "-outacti":
                output_a = parse.get_output_acti_arg(argv, index)
            if mod == "-opti":
                optim = parse.get_opti(argv, index)
            if mod == "-early":
                (early, patience) = parse.get_early(argv, index, patience)
            if mod == "-multi":
                (multi, cases) = parse.get_multi(argv, index, cases)
            if mod == "-display":
                display = parse.get_display(argv, index)

    # load data
    x = load_data("data/x_train.csv")
    y = load_data("data/y_train.csv")
    x_val = load_data("data/x_val.csv")
    y_val = load_data("data/y_val.csv")

    # apply min max or zscore normalization
    my_Scaler = Minmax_Scaler()
    # my_Scaler = Standard_Scaler()
    my_Scaler.fit(x)
    X = my_Scaler.transform(x)
    X_val = my_Scaler.transform(x_val)
    # save scaler to reuse later on the test set
    joblib.dump(my_Scaler, 'scaler.pkl')

    # #  transpose matrices
    X = X.T
    y = y.reshape((1, y.shape[0]))
    X_val = X_val.T
    y_val = y_val.reshape((1, y_val.shape[0]))

    # apply training model
    if multi == "yes":
        plt.figure(figsize=(8, 8))
        for i in cases[0]:
            for j in cases[1]:
                for k in cases[2]:
                    for m in cases[3]:
                        for p in cases[4]:
                            model = network(hidden_layers=i, lr=j, n_epochs=1000, init=init,
                                            hidden_a=k, output_a=m, optim=p, early=early,
                                            patience=patience, multi=multi)
                            model.neural_network(X, y, X_val, y_val)
        plt.title("Validation losses")
        plt.grid(True, linestyle='--', alpha=0.5)
    else:
        model = network(hidden_layers=hidden_layers, lr=lr, n_epochs=n_epochs,
                        init=init, hidden_a=hidden_a, output_a=output_a, optim=optim,
                        early=early, patience=patience, display=display)
        model.neural_network(X, y, X_val, y_val)

    plt.show()

    # except Exception as e:
    #     print(e)

# ./train.py -lr 0.3 -epochs 3000
# ./train.py -hidden 32 16 -lr 0.3 -epochs 1500 -outacti softmax
