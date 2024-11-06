def helper_train():
    """
Usage:
    training.py [-h] [-hidden LAYERS] [-epochs EPOCHS] [-init INIT] [-lr LEARNINGRATE] [-hidacti HIDACTI] [-outacti OUTACTI] [-opti OPTI] [-early EARLY] [-multi OUTACTI]

Options:
    -h, --help					Show this help message and exit.
    -hidden LAYERS					Neurons number on each hidden layer.
    -epochs EPOCHS					Number of epochs applied on the network.
    -lr LEARNING RATE				Learning rate of the model.
    -init INIT					Initialization mode of the neural network (random or xavier).
    -hidacti HIDACTI				Hidden layers activation function (sigmoid, tanh, relu or leaky).
    -outacti OUTACTI				Output layer activation function (sigmoid, softmax1 or softmax).
    -opti OPTI					Optimization model (mom, rms or adam).
    -early                      Set early stopping (yes or no) with patience as second optional prm (default is 10)
    -multi                      ask for multiple trainings
    -display                    Display training at each iteration (yes or no)
    """
    pass


def helper_prediction():
    """
Usage:
    prediction.py [-h] [MODEL]

Options:
    -h, --help					Show this help message and exit.
    MODEL						Saved model .pkl file.
    """
    pass
