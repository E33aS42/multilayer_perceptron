#!/usr/bin/env python3


class parser():
    def __init__(self):
        self.options = ["-epochs", "-lr", "-init", "-hidacti",
                        "-outacti", "-opti", "-early", "-multi", "-display"]


    def is_integer(self, s):
        """
        This function checks if a string can be converted to an integer.
        """
        try:
            int(s)
            return True
        except Exception:
            return False


    def is_float(self, s):
        """
        This function checks if a string can be converted to a float.
        """
        try:
            float(s)
            return True
        except ValueError:
            return False


    def get_hidden_arg(self, argv, index):
        try:
            args = argv[index + 1:]
            assert self.is_integer(args[0])\
                and int(args[0]) > 0\
                and self.is_integer(
                args[1]) and int(args[1]) > 0, \
                "network should have at least 2 hidden layers"
            layers = []
            for a in args:
                if not self.is_integer(a):
                    print(self.is_integer(a))
                    break
                layers.append(int(a))
            return tuple(layers)

        except Exception as e:
            print(e)
            exit()


    def get_epochs_arg(self, argv, index):
        try:
            assert self.is_integer(
                argv[index + 1]) and int(argv[index + 1]) > 0, \
                "input is not a positive integer"
            return int(argv[index + 1])

        except Exception as e:
            print(e)
            exit()


    def get_lr_arg(self, argv, index):
        try:
            assert self.is_float(
                argv[index + 1]) and float(argv[index + 1]) > 0, \
                "input is not a positive float"
            return float(argv[index + 1])

        except Exception as e:
            print(e)
            exit()


    def get_init_arg(self, argv, index):
        try:
            assert argv[index + 1] in ["random",
                                       "xavier"], \
                "unknown initialization method"
            return argv[index + 1]

        except Exception as e:
            print(e)
            exit()


    def get_hidden_acti_arg(self, argv, index):
        try:
            assert argv[index + 1] in ["sigmoid", "tanh", "relu",
                                       "leaky"], \
                "unknown hidden layer activation method"
            return argv[index + 1]

        except Exception as e:
            print(e)
            exit()


    def get_output_acti_arg(self, argv, index):
        try:
            assert argv[index + 1] in ["sigmoid", "softmax",
                                       "softmax1"], \
                "unknown output layer activation method"
            return argv[index + 1]

        except Exception as e:
            print(e)
            exit()


    def get_opti(self, argv, index):
        try:
            assert argv[index + 1] in ["none", "mom", "rms", "RMS",
                                       "adam", "nesterov"], \
                "unknown opimization function"
            # check if betas parameters are present
            # otherwise default prms will be used
            if len(argv) > index + 3 and self.is_float(argv[index + 2])\
                    and self.is_float(argv[index + 3])\
                    and float(argv[index + 2]) > 0 and float(argv[index + 2]) <= 1\
                    and float(argv[index + 3]) > 0 and float(argv[index + 3]) <= 1:
                return argv[index + 1], float(argv[index + 2]), float(argv[index + 3])
            elif len(argv) > index + 2 and self.is_float(argv[index + 2])\
                    and float(argv[index + 2]) > 0 and float(argv[index + 2]) <= 1:
                return argv[index + 1], float(argv[index + 2]), 0.999
            else:
                return argv[index + 1], 0.9, 0.999

        except Exception as e:
            print(e)
            exit()


    def get_early(self, argv, index, patience):
        try:
            assert argv[index + 1] in ["no", "yes"], \
                "setting early stopping by inputing yes or no"
            if len(argv) > index + 2 and argv[index + 2] not in self.options\
                    and self.is_integer(argv[index + 2])\
                    and int(argv[index + 2]) > 1:
                return argv[index + 1], eval(argv[index + 2])
            else:
                return argv[index + 1], patience

        except Exception as e:
            print(e)
            exit()


    def get_multi(self, argv, index, cases):
        try:
            assert argv[index + 1] in ["no", "yes"], \
                "display multiple learning curves by inputing no or yes"
            if len(argv) > index + 2 and argv[index + 2] not in self.options\
                    and isinstance(eval(argv[index + 2]), list)\
                    and len(eval(argv[index + 2])) == 5:
                return argv[index + 1], eval(argv[index + 2])
            else:
                return argv[index + 1], cases

        except Exception as e:
            print(e)
            exit()


    def get_display(self, argv, index):
        try:
            assert argv[index + 1] in ["no",
                                       "yes"], \
                "setting display mode by inputing yes or no"
            return argv[index + 1]

        except Exception as e:
            print(e)
            exit()
