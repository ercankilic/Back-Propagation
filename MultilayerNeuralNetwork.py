from NeuralNetwork import NeuralNetwork as nn
from itertools import count
import numpy as np

class MultilayerNeuralNetwork:

    def __init__(self, num_of_inputs, num_of_hidden_layers, hidden_layer_neurons_num, num_of_outputs):

        self.num_of_inputs = num_of_inputs
        self.num_of_hidden_layers = num_of_hidden_layers
        self.num_of_outputs = num_of_outputs
        self.hidden_layer_neurons_num = hidden_layer_neurons_num

        self.hidden_layers = []
        self.init_hidden_layers()

        self.output_layer = None
        self.init_output_layer()

        self.hidden_layers.append(self.output_layer)
        del self.output_layer

    def init_hidden_layers(self):
        if self.num_of_hidden_layers != 0:
            # First hidden layer
            self.hidden_layers.append(nn(self.hidden_layer_neurons_num[0], self.num_of_inputs))
            for i in range(self.num_of_hidden_layers - 1):
                # Other hidden layers
                self.hidden_layers.append(nn(self.hidden_layer_neurons_num[i + 1], self.hidden_layer_neurons_num[i]))

    def init_output_layer(self):
        if self.num_of_hidden_layers != 0:
            self.output_layer = nn(self.num_of_outputs, self.hidden_layer_neurons_num[-1])
        else:
            self.output_layer = nn(self.num_of_outputs, self.num_of_inputs)

    def get_output(self, inputs):
        propagate_input = []
        propagate_input = np.copy(inputs)
        temp_input = []
        if self.hidden_layers is not None:
            for hy in self.hidden_layers:
                temp_input = hy.get_outputs(propagate_input)
                propagate_input = []
                propagate_input = np.copy(temp_input)
                temp_input.clear()

        self.output = propagate_input
        return self.output

    def get_error(self, t_outputs):
        error = self.hidden_layers[-1].get_error(t_outputs)
        return error

    def train(self, inputs, t_outputs, epoch=500):
        self.error = []
        for i in range(epoch):
            err = 0
            for input, target in zip(inputs, t_outputs):
                self._train(input, target)
                r = self.get_output(input)
                err += self.get_error(target)
                print("%.2f" %(100 * i / epoch), target.index(1), np.argmax(r))
            self.error.append((err / len(inputs)))
        return self.error

    def _train(self, inputs, t_output):
        self.get_output(inputs)

        # From output to input layer throughout hidden layers
        total_neurons =[]
        total_neurons.extend(self.hidden_layer_neurons_num)
        total_neurons.append(self.num_of_outputs)
        for k, layer, num in zip(count(), reversed(self.hidden_layers), reversed(total_neurons)):
            for i in range(num): # Get neuron of each layer to get derivatives of error
                temp_neuron = layer.get_neuron(i)
                # If neuron is in one of the hidden layers, it should gets ro of next layer for derivative error by out
                if k > 0:
                    d_error_by_out = self.next_layer_er_by_out(k, i, total_neurons)
                else: # If neuron is in the output layer, it directly calculates this value
                    d_error_by_out = temp_neuron.derivative_error_by_out(t_output[i])
                temp_neuron.set_d_error_by_out(d_error_by_out)
                temp_neuron.set_d_out_by_net(temp_neuron.derivative_out_by_net())
                temp_neuron.set_ro()
                for j in range(temp_neuron.get_number_of_weight()): # Get each weight of that neuron to setting up to new values
                    d_net_by_weight = temp_neuron.derivative_net_by_weight(j)
                    d_error_by_weight = temp_neuron.get_ro() * d_net_by_weight
                    temp_neuron.set_weight(j, d_error_by_weight)

    def next_layer_er_by_out(self, k, i, total_neurons):
        layer = self.hidden_layers[-k]
        neuron_num = total_neurons[-k]
        d_error_by_out = 0
        for j in range(neuron_num):
            neuron = layer.get_neuron(j)
            d_error_by_out += (neuron.get_ro() * neuron.get_weight(i))

        return d_error_by_out
