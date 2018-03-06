from MultilayerNeuralNetwork import MultilayerNeuralNetwork as mnn
from FileIO import FileIO as fio
import numpy as np
from matplotlib import pyplot as plt

def prepare_data(proportion, instance_num, class_num):
    train, test = [], []
    train_target, test_target = [], []
    class_instance_num = int(instance_num / class_num)
    num = class_instance_num * proportion / 100
    c = [f.data[0][-1]]
    for i in range(class_num):
        data = f.data[i * class_instance_num:class_instance_num * (i + 1)]
        for j, l in enumerate(data):
            t = [0] * class_num
            if j >= num:
                train.append(l[:-1])
                if l[-1] in c:
                    t[c.index(l[-1])] = 1
                    train_target.append(t)
                else:
                    c.append(l[-1])
                    t[c.index(l[-1])] = 1
                    train_target.append(t)
            else:
                test.append(l[:-1])
                if l[-1] in c:
                    t[c.index(l[-1])] = 1
                    test_target.append(t)
                else:
                    c.append(l[-1])
                    t[c.index(l[-1])] = 1
                    test_target.append(t)
    return [np.array(train, dtype="float64"), train_target, np.array(test, dtype="float64"), test_target]

path = "./dataset/iris_data.txt"
delimiter = ","

f = fio(path, delimiter)

f.read_file()

[train, train_target, test, test_target] = prepare_data(30, 150, 3)

# for i, j in zip(train, train_target):
#     print(i, "->", j)
#
# print("------------------------")
#
# for i, j in zip(test, test_target):
#     print(i, "->", j)

# for l in f.data:
#     print(l)

hidden_layer_num = 1
hidden_layer_neuron_num = [5] * hidden_layer_num
net = mnn(4, hidden_layer_num, hidden_layer_neuron_num, 3)

epoch = 500

err = net.train(train, train_target, epoch)
acc = 0
for input, target in zip(test, test_target):
    r = net.get_output(input)
    if target.index(1) == np.argmax(r):
        acc += 1
    print(target.index(1), np.argmax(r))

print("%.2f" %(100 * acc / len(test)))

plt.plot(err)
plt.xlabel("Epoch")
plt.ylabel("Mean Square Error")
plt.show()
