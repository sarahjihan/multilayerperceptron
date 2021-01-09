from os import listdir
from mlpmat import MLP
from image import ImageReader
import numpy as np

# mlp layer: 784 nodes (input) -> 15 nodes (hidden) -> 10 nodes (output)
mlp = MLP([784, 15, 10], learning_rate = 0.001, momentum = 0.05, mini_batch = 100, print_error_progress = True)

datafilename = []
len_section = []

for i in range(10):
    temp = []
    for f in listdir('../data/mnist/' + str(i)):
        if f[0] == '.':
            continue
        else:
            temp.append('../data/mnist/' + str(i) + '/' + f)
    len_section.append(len(temp))
    datafilename.append(temp)

def all_decept():
    for l in len_section:
        if l > 0:
            return False

    return True

def loadDataset():
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    print "Loading data set..."
    while not all_decept() and count != [501, 501, 501, 501, 501, 501, 501, 501, 501, 501]:
        for j in range(10):
            expextation = [
                1 if j == 0 else 0,
                1 if j == 1 else 0,
                1 if j == 2 else 0,
                1 if j == 3 else 0,
                1 if j == 4 else 0,
                1 if j == 5 else 0,
                1 if j == 6 else 0,
                1 if j == 7 else 0,
                1 if j == 8 else 0,
                1 if j == 9 else 0
            ]
            if count[j] > 500:
                continue
            if len_section[j] > 0:
                # print "load: " + str(count[j]) + ": " + datafilename[j][len_section[j] - 1] + " => " + str(expextation)
                mlp.TrainData.append([np.array(ImageReader().read(datafilename[j][len_section[j] - 1])) / 255, expextation])
                len_section[j] -= 1
                count[j] += 1

    print "Dataset loaded!"

loadDataset()
mlp.train(100)
mlp.saveModel("model/mnist784-15-10")
