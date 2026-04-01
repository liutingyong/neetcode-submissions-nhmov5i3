import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        forward = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-forward))
        loss = 0.5 * (y_hat - y_true) ** 2
        weight_gradients = (y_hat - y_true) * y_hat * (1 - y_hat)
        dL_dw = np.round(weight_gradients * x, 5)
        dL_db = round(float(weight_gradients), 5)
        return (dL_dw, dL_db)
