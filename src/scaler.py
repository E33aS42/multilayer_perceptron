import numpy as np


class Minmax_Scaler():
    def __init__(self):
        pass

    def fit(self, x):
        try:
            assert isinstance(x, np.ndarray) and (
                x.ndim >= 1), "argument must be a numpy.ndarray,\
                    a vector of dimension m * n"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            m, n = x.shape
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            self.minx = np.min(x, axis=0)
            self.maxx = np.max(x, axis=0)

        except Exception as e:
            print(e)
            return None

    def transform(self, x):
        try:
            assert isinstance(x, np.ndarray) and (
                x.ndim >= 1), "argument must be a numpy.ndarray,\
                    a vector of dimension m * n"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            m, n = x.shape
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            assert len(self.minx) == x.shape[1] and len(
                self.maxx) == x.shape[1], "Wrong dimension of numpy.ndarray"
            x_ = (x - self.minx) / (self.maxx - self.minx)
            return x_

        except Exception as e:
            print(e)
            return None

    def de_minmax(self, x):
        """Denormalize a previously normalized non-empty numpy.ndarray\
            using the min-max standardization.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        x_ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
        This function shouldnt raise any Exception.
        """
        try:
            assert isinstance(x, np.ndarray) and (
                x.ndim >= 1), "argument must be a numpy.ndarray,\
                    a vector of dimension m * n"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            m, n = x.shape
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            assert len(self.minx) == x.shape[1] and len(
                self.maxx) == x.shape[1], "Wrong dimension of numpy.ndarray"
            x_ = x * (self.maxx - self.minx) + self.minx
            return x_

        except Exception as e:
            print(e)
            return None


class Standard_Scaler():
    def __init__(self):
        pass

    def fit(self, x):
        try:
            assert isinstance(x, np.ndarray) and (
                x.ndim >= 1), "argument must be a numpy.ndarray,\
                    a vector of dimension m * n"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            self.mu = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)

        except Exception as e:
            print(e)
            return None

    def transform(self, x):
        try:
            assert isinstance(x, np.ndarray) and (
                x.ndim >= 1), "argument must be a numpy.ndarray,\
                    a vector of dimension m * n"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            assert len(self.std) == x.shape[1] and len(
                self.mu) == x.shape[1], "wrong dimension of numpy.ndarray"
            x_ = x - self.mu
            x_ /= self.std
            return x_
        except Exception as e:
            print(e)
            return None

    def de_zscore(self, x):
        """Denormalize a previously normalized non-empty numpy.ndarray\
            using the min-max standardization.
        Args:
        x: has to be an numpy.ndarray, a vector.
        Returns:
        x_ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
        Raises:
        This function shouldnt raise any Exception.
        """
        try:
            assert isinstance(x, np.ndarray) and (
                x.ndim >= 1), "argument must be a numpy.ndarray,\
                    a vector of dimension m * n"
            assert np.any(x), "argument cannot be an empty numpy.ndarray"
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            assert len(self.std) == x.shape[1] and len(
                self.mu) == x.shape[1], "wrong dimension of numpy.ndarray"
            x_ = x * self.std + self.mu
            return x_

        except Exception as e:
            print(e)
            return None
