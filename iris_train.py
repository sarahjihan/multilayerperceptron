import numpy as np
from mlpmat import MLP

len_output_map = 0
output_map = {}
inv_output_map = []

def normalize(self, datas, offset):
	out = []
	for i in range(len(datas)):
		if self.maxdata[offset + i] - self.mindata[offset + i] == 0:
			out.append(datas[i])
		else:
			out.append((datas[i] - self.mindata[offset + i]) / (self.maxdata[offset + i] - self.mindata[offset + i]))

	return out

def createOutputArray(n):
	out = [0] * len_output_map
	out[int(n)] = 1

	return out

def normalizeIrisData(iris_data):

	global output_map
	global len_output_map

	len_data = len(iris_data[0])
	maxdata = iris_data[0][:-1]
	mindata = iris_data[0][:-1]

	print "Normalizing: get min max data..."
	for idata in iris_data:
		for i in range(len_data):
			if i != (len_data - 1):
				maxdata[i] = idata[i] if idata[i] > maxdata[i] else maxdata[i]
				mindata[i] = idata[i] if idata[i] < mindata[i] else mindata[i]
			else:
				if not idata[i] in output_map:
					inv_output_map.append(len(output_map))
					output_map[idata[i]] = len(output_map)

	len_output_map = len(output_map)

	for idata in iris_data:
		for i in range(len_data - 1):
			divider = maxdata[i] - mindata[i]
			if divider > 0:
				idata[i] = (idata[i] - mindata[i]) / divider

	last_index = len_data - 1
	for iris in iris_data:
		iris[last_index] = createOutputArray(output_map[iris[last_index]])

	return iris_data

def readfiles(filename):
	f = open(filename)
	content = f.read().splitlines()
	data = []
	i = 0
	for i in range(len(content)):
		if i == 0: continue
		dataKeI = content[i].split(",")
		for j in range(len(dataKeI) - 1):
			dataKeI[j] = float(dataKeI[j])
		data.append(dataKeI)

	return data

def simpulkan(arr):
    # print arr
    maxIdx = 0
    for i in range(len(arr)):
        if arr[i] > arr[maxIdx]:
            maxIdx = i

    return maxIdx

irisdata = readfiles("iris_data/iris.txt")
normalized_data = normalizeIrisData(irisdata)
# print normalized_data

# mlp layer: 784 nodes (input) -> 15 nodes (hidden) -> 10 nodes (output)
mlp = MLP([4, 7, 3], learning_rate = 0.01, momentum = 0.05, print_error_progress = True, print_testing_detail = True, full_testing_periode = 100)

for data in normalized_data:
	mlp.TrainData.append([data[:-1], data[-1:][0]])
mlp.loadModel("model/iris")

mlp.train(10000)

mlp.saveModel("model/iris")
