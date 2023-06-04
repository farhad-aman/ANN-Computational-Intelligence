import numpy as np
from numpy import ndarray


class BinaryCrossEntropy:
    def __init__(self) -> None:
        pass

    def compute(self, y_hat: np.ndarray, y: np.ndarray) -> ndarray:
        """
        Computes the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                binary cross entropy loss
        """
        batch_size = y.shape[1]
        cost = (-1/batch_size) * (np.dot(y, np.log(y_hat).T) + np.dot((1-y), np.log(1-y_hat).T))
        return np.squeeze(cost)

    def backward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the binary cross entropy loss.
            args:
                y: true labels (n_classes, batch_size)
                y_hat: predicted labels (n_classes, batch_size)
            returns:
                derivative of the binary cross entropy loss
        """
        return - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))

