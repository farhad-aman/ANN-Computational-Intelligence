import pickle
import numpy as np
from tqdm import tqdm

# layers
from layers.maxpooling2d import MaxPool2D
from layers.convolution2d import Conv2D
from layers.fullyconnected import FC

# activations
from activations import ReLU
from activations import Sigmoid
from activations import LinearActivation
from activations import Tanh


class Model:
    def __init__(self, arch, criterion, optimizer, name=None):
        """
        Initialize the model.
        args:
            arch: dictionary containing the architecture of the model
            criterion: loss 
            optimizer: optimizer
            name: name of the model
        """
        if name is None:
            self.model = arch
            self.criterion = criterion
            self.optimizer = optimizer
            self.layers_names = list(arch.keys())
        else:
            self.model, self.criterion, self.optimizer, self.layers_names = self.load_model(name)

    def is_layer(self, layer):
        """
        Check if the layer is a layer.
        args:
            layer: layer to be checked
        returns:
            True if the layer is a layer, False otherwise
        """
        return type(layer) == MaxPool2D or type(layer) == Conv2D or type(layer) == FC

    def is_activation(self, layer):
        """
        Check if the layer is an activation function.
        args:
            layer: layer to be checked
        returns:
            True if the layer is an activation function, False otherwise
        """
        return type(layer) == Sigmoid or type(layer) == LinearActivation or type(layer) == ReLU or type(layer) == Tanh

    def forward(self, x):
        """
        Forward pass through the model.
        args:
            x: input to the model
        returns:
            output of the model
        """
        tmp = []
        A = x
        for i in range(0, len(self.layers_names), 2):
            Z = self.model[self.layers_names[i]].forward(A)
            tmp.append(np.copy(Z))
            A = self.model[self.layers_names[i + 1]].forward(Z)
            tmp.append(np.copy(A))
        return tmp

    def backward(self, dAL, tmp, x):
        """
        Backward pass through the model.
        args:
            dAL: derivative of the cost with respect to the output of the model
            tmp: list containing the intermediate values of Z and A
            x: input to the model
        returns:
            gradients of the model
        """
        dA = dAL
        grads = {}
        for i in range(len(tmp) - 1, -1, -2):
            if i > 2:
                Z, A = tmp[i - 1], tmp[i - 2]
            else:
                Z, A = tmp[i - 1], x
            dZ = self.model[self.layers_names[i]].backward(dA, Z)
            dA, grad = self.model[self.layers_names[i - 1]].backward(dZ, A)
            grads[self.layers_names[i - 1]] = grad
        return grads

    def update(self, grads):
        """
        Update the model.
        args:
            grads: gradients of the model
        """
        for name in self.layers_names:
            if self.is_layer(self.model[name]) and not type(self.model[name]) == MaxPool2D:
                self.model[name].update_parameters(self.optimizer, grads[name])

    def one_epoch(self, x, y):
        """
        One epoch of training.
        args:
            x: input to the model
            y: labels
            batch_size: batch size
        returns:
            loss
        """
        tmp = self.forward(x)
        AL = tmp[-1]
        loss = self.criterion.compute(AL, y)
        dAL = self.criterion.backward(AL, y)
        grads = self.backward(dAL, tmp, x)
        self.update(grads)
        return loss

    def save(self, name):
        """
        Save the model.
        args:
            name: name of the model
        """
        with open(name, 'wb') as f:
            pickle.dump((self.model, self.criterion, self.optimizer, self.layers_names), f)

    def load_model(self, name):
        """
        Load the model.
        args:
            name: name of the model
        returns:
            model, criterion, optimizer, layers_names
        """
        with open(name, 'rb') as f:
            return pickle.load(f)

    def shuffle(self, m, shuffling):
        order = list(range(m))
        if shuffling:
            return np.random.shuffle(order)
        return order

    def batch(self, X, y, batch_size, index, order):
        """
        Get a batch of data.
        args:
            X: input to the model
            y: labels
            batch_size: batch size
            index: index of the batch
                e.g: if batch_size = 3 and index = 1 then the batch will be from index [3, 4, 5]
            order: order of the data
        returns:
            bx, by: batch of data
        """
        last_index = index * batch_size
        batch = order[last_index: last_index + batch_size]
        if X.ndim == 2:
            bx = X[:, np.asarray(batch)]
            by = y[:, np.asarray(batch)]
            return bx, by
        else:
            bx = X[np.asarray(batch), :, :, :]
            by = y[np.asarray(batch)]
            return bx, by

    def compute_loss(self, X, y, batch_size):
        """
        Compute the loss.
        args:
            X: input to the model
            y: labels
            Batch_Size: batch size
        returns:
            loss
        """
        m = X.shape[0] if X.ndim == 4 else X.shape[1]
        order = self.shuffle(m, False)
        cost = 0
        for b in range(m // batch_size):
            bx, by = self.batch(X, y, batch_size, b, order)
            tmp = self.forward(bx)
            AL = tmp[-1]
            cost += self.criterion.compute_cost(AL, y)(m // batch_size)
        return cost

    def train(self, X, y, epochs, val=None, batch_size=3, shuffling=False, verbose=1, save_after=None):
        """
        Train the model.
        args:
            X: input to the model
            y: labels
            epochs: number of epochs
            val: validation data
            batch_size: batch size
            shuffling: if True shuffle the data
            verbose: if 1 print the loss after each epoch
            save_after: save the model after training
        """
        train_cost = []
        val_cost = []
        m = X.shape[0] if X.ndim == 4 else X.shape[1]
        for e in tqdm(range(1, epochs + 1)):
            order = self.shuffle(m, shuffling)
            cost = 0
            for b in range(m // batch_size):
                bx, by = self.batch(X, y, batch_size, b, order)
                cost += self.one_epoch(bx, by) / (m // batch_size)
            train_cost.append(cost)
            if val is not None:
                val_cost.append(self.compute_loss(X, y, batch_size))
            if verbose:
                if e % verbose == 0:
                    print("Epoch {}: train cost = {}".format(e, cost))
                if val is not None:
                    print("Epoch {}: val cost = {}".format(e, val_cost[-1]))
        if save_after is not None:
            self.save(save_after)
        return train_cost, val_cost

    def predict(self, X):
        """
        Predict the output of the model.
        args:
            X: input to the model
        returns:
            predictions
        """
        A0 = X
        AL = self.forward(A0)[-1]
        return AL
