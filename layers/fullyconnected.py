import numpy as np


class FC:
    def __init__(self, input_size: int, output_size: int, name: str, initialize_method: str = "random"):
        self.input_size = input_size
        self.output_size = output_size
        self.name = name
        self.initialize_method = initialize_method
        self.parameters = [self.initialize_weights(), self.initialize_bias()]
        self.input_shape = None
        self.reshaped_shape = None

    def initialize_weights(self):
        if self.initialize_method == "random":
            return np.random.randn(self.output_size, self.input_size)
        elif self.initialize_method == "xavier":
            return np.random.randn(self.output_size, self.input_size) * np.sqrt(1 / self.input_size)
        elif self.initialize_method == "he":
            return np.random.randn(self.output_size, self.input_size) * np.sqrt(2 / self.input_size)
        else:
            raise ValueError("Invalid initialization method")

    def initialize_bias(self):
        return np.zeros((self.output_size, 1))

    def forward(self, A_prev):
        """
        Forward pass for fully connected layer.
            args:
                A_prev: activations from previous layer (or input data)
                A_prev.shape = (batch_size, input_size)
            returns:
                Z: output of the fully connected layer
        """
        self.input_shape = A_prev.shape
        A_prev_tmp = np.copy(A_prev)
        if A_prev.ndim == 4:
            batch_size = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
        self.reshaped_shape = A_prev_tmp.shape
        W, b = self.parameters[0], self.parameters[1]
        Z = W @ A_prev_tmp + b
        return Z

    def backward(self, dZ, A_prev):
        """
        Backward pass for fully connected layer.
            args:
                dZ: derivative of the cost with respect to the output of the current layer
                A_prev: activations from previous layer (or input data)
            returns:
                dA_prev: derivative of the cost with respect to the activation of the previous layer
                grads: list of gradients for the weights and bias
        """
        A_prev_tmp = np.copy(A_prev)
        if A_prev.ndim == 4:
            batch_size = A_prev.shape[0]
            A_prev_tmp = A_prev_tmp.reshape(batch_size, -1).T
        W, b = self.parameters[0], self.parameters[1]
        dW = (1 / A_prev.shape[1]) * np.dot(dZ, A_prev_tmp.T)
        db = (1 / A_prev.shape[1]) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)
        grads = [dW, db]
        if len(self.input_shape) == 4:
            dA_prev = dA_prev.T.reshape(self.input_shape)
        return dA_prev, grads

    def update_parameters(self, optimizer, grads):
        """
        Update the parameters of the layer.
            args:
                optimizer: optimizer object
                grads: list of gradients for the weights and bias
        """
        self.parameters = optimizer.update(grads, self.name)
