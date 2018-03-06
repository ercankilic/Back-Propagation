import numpy as np
import random as rd
import OptionalParameters as OP

class Neuron:

    def __init__(self,num_of_inputs):

        self.num_of_inputs = num_of_inputs
        self.weight = []
        self.bias = 0
        self.init_bias(OP.weight_init_type)
        self.init_weight(OP.weight_init_type)
        self.init_learning_rate(OP.learning_rate)
        self.d_error_by_out = 1
        self.d_out_by_net = 1
        self.ro = 1
        self.prev_d_error_by_weight = 1
        self.delta_weight = [0] * self.num_of_inputs

    def init_bias(self, weight_init_type):
        if weight_init_type == "random":
            self.bias = rd.random()
        elif weight_init_type == "zero":
            self.bias = 0.0

    def init_weight(self, weight_init_type):
        if weight_init_type == "random":
            self.weight = np.array([rd.random() for _ in range(self.num_of_inputs)])
        elif weight_init_type == "zero":
            self.weight = np.array([0.0 for _ in range(self.num_of_inputs)])

    def init_learning_rate(self, learning_rate):
        if OP.backprop_alg == "delta-bar" or OP.backprop_alg == "adaptive":
            OP.learning_rate = [learning_rate] * self.num_of_inputs
        else:
            OP.learning_rate = learning_rate

    def set_ro(self):
        self.ro = self.d_error_by_out * self.d_out_by_net

    def get_ro(self):
        return self.ro

    def get_weight(self, index):
        return self.weight[index]

    def set_weight(self, index, d_error_by_weight):
        prev_delta_weight = self.delta_weight[index]
        if OP.backprop_alg == "backprop":
            self.delta_weight[index] = -(OP.learning_rate * d_error_by_weight)
            self.weight[index] += self.delta_weight[index]
        elif OP.backprop_alg == "momentum":
            self.delta_weight[index] = -(OP.learning_rate * d_error_by_weight)
            self.weight[index] += (self.delta_weight[index] + OP.momentum * prev_delta_weight)
        elif OP.backprop_alg == "delta-bar":
            self.delta_weight[index] = -(OP.learning_rate[index] * d_error_by_weight)
            product = self.prev_d_error_by_weight * d_error_by_weight
            OP.learning_rate[index] = OP.learning_rate[index] + OP.kappa if product > 0 \
                else ((1 - OP.gamma) * OP.learning_rate[index]
                      if product < 0 else OP.learning_rate[index])
            self.weight[index] += OP.learning_rate[index] * self.delta_weight[index]
            self.prev_d_error_by_weight = (1 - OP.beta) * d_error_by_weight + OP.beta * self.prev_d_error_by_weight
        elif OP.backprop_alg == "adaptive":
            OP.learning_rate[index] = OP.eta if OP.learning_rate[index] >= OP.eta else \
                                    prev_delta_weight / (d_error_by_weight + prev_delta_weight / OP.learning_rate[index])
            self.delta_weight[index] = -(OP.learning_rate[index] * d_error_by_weight)
            self.weight[index] += self.delta_weight[index]

    def get_number_of_weight(self):
        return self.num_of_inputs

    def get_bias(self):
        return self.bias

    def set_bias(self, d_error_by_weight):
        self.bias -= (OP.learning_rate * d_error_by_weight)

    def get_neuron_output(self, input):
        self.input = input
        y_in = np.dot(self.input, self.weight) + self.bias
        self.output = self.activate(y_in)
        return self.output

    def activate(self, y_in, beta=1.0):
        if OP.activation_func == "sigmoid":
            return 1 / (1 + np.exp( -1 * beta * y_in))
        elif OP.activation_func == "tanh":
            return (2 / (1 + np.exp(-2 * y_in))) - 1
        elif OP.activation_func == "relu":
            return 0 if y_in < 0 else y_in

    # Mean Square Error Value
    def get_error(self, t_output):
        return 0.5 * (t_output - self.output) ** 2

    # d (E total) / d (output)
    def derivative_error_by_out(self, t_output):
        return self.output - t_output

    # d (output) / d (net)
    def derivative_out_by_net(self):
        if OP.activation_func == "sigmoid":
            return self.output * (1 - self.output)
        elif OP.activation_func == "tanh":
            return 1 - ((2 / (1 + np.exp(-2 * self.output))) - 1) ** 2
        elif OP.activation_func == "relu":
            return 0 if self.output < 0 else 1

    # d (net) / d (weight)
    def derivative_net_by_weight(self, weight_index):
        return self.input[weight_index]

    def set_d_error_by_out(self, value):
        self.d_error_by_out = value

    def get_d_error_by_out(self):
        return self.d_error_by_out

    def set_d_out_by_net(self, value):
        self.d_out_by_net = value

    def get_d_out_by_net(self):
        return self.d_out_by_net
