from Neuron import Neuron

class NeuralNetwork:

    def __init__(self, num_neurons, num_of_inputs):
        self.neurons = []
        self.num_neurons = num_neurons
        for _ in range(self.num_neurons):
            self.neurons.append(Neuron(num_of_inputs))

    def get_neuron(self, index):
        return self.neurons[index]

    def get_outputs(self, input):
        self.outputs = []
        for n in self.neurons:
            self.outputs.append(n.get_neuron_output(input))
        return self.outputs

    def get_stored_output(self):
        return self.outputs

    def get_error(self, t_outputs):
        error = 0
        for n, t in zip(self.neurons, t_outputs):
            error += n.get_error(t)
        return error / len(t_outputs)
