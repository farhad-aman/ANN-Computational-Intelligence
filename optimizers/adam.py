import numpy as np


class Adam:
    def __init__(self, layers_list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.V = {}
        self.S = {}
        for name in layers_list:
            v = [np.zeros_like(p) for p in layers_list[name].parameters]
            s = [np.zeros_like(p) for p in layers_list[name].parameters]
            self.V[name] = v
            self.S[name] = s

    def update(self, grads, name, epoch):
        layer = self.layers[name]
        params = []
        for i in range(len(grads)):
            self.V[name][i] = self.beta1 * self.V[name][i] + (1 - self.beta1) * grads[i]
            self.S[name][i] = self.beta2 * self.S[name][i] + (1 - self.beta2) * np.power(grads[i, 2])
            self.V[name][i] /= 1 - self.beta1 ** epoch
            self.S[name][i] /= 1 - self.beta2 ** epoch
            params.append(layer.parameters[i] - self.learning_rate * self.V[name][i] / np.sqrt(
                self.S[name][i] + self.epsilon))
        return params
