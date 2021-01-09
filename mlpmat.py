import random
import numpy as np

class MLP:

	matWeight	= []
	matWeightE	= []
	matNeuronL 	= []
	miniBatchW	= []
	miniBatchB	= []
	miniBatchN	= 1
	matWBias	= []
	matWBiasE	= []
	NLayerCount = []
	TrainData	= []
	LearningRte = 0.9
	Momentum 	= 0.1
	biasNodeValue = 1
	biasInitWeigt = 0.5
	testingPeriode = 100
	printBatchProgress = False
	printErrorProgress = False
	printTestingDetail = False
	maxdata 	= []
	mindata		= []
	normalized  = False 
	modelLoadedDoesntMatchLayer = False

	def __init__(self, layers, learning_rate = 0.5, momentum = 0, mini_batch = 1, print_batch_progress = False, full_testing_periode = 1, print_testing_detail = False, print_error_progress = False):
		for layerN in layers:
			self.NLayerCount.append(layerN)

		self.LearningRte = learning_rate
		self.Momentum = momentum
		self.miniBatchN = mini_batch
		self.testingPeriode = full_testing_periode
		self.printBatchProgress = print_batch_progress
		self.printErrorProgress = print_error_progress
		self.printTestingDetail = print_testing_detail
		self.initMatWeights()
		self.initMatLayer()

	def sigmoid(self, x, deriv = False):
	    if deriv:
	        return x * (1 - x)

	    return 1 / (1 + np.exp(-x))

	def saveModel(self, name):
		np.save(name + "-w", self.matWeight)
		np.save(name + "-b", self.matWBias)

	def getNumberOfWeightOnModel(self, model):
		out = []
		outputLayer = 0
		for n in model:
			out.append(len(n))
			outputLayer = len(n[0])
		out.append(outputLayer)

		return out

	def getNumberOfBiasOnModel(self, model):
		out = []
		for n in model:
			# TODO: needs some refactor
			if type(n[0]) is np.float64 or type(n[0]) is np.int64:
				out.append(len(n))
			else:
				out.append(len(n[0]))
				
		return out

	def loadModel(self, name):
		weightLoaded = np.load(name + "-w.npy")
		wBiasLoaded  = np.load(name + "-b.npy")
 
		a = self.getNumberOfWeightOnModel(weightLoaded)
		b = self.getNumberOfWeightOnModel(self.matWeight)
		if a != b:
			print "Error #1: your weight model \"", name, "\" has", a, "layer does not match with current weight model", b
			exit()

		a = self.getNumberOfBiasOnModel(wBiasLoaded)
		b = self.getNumberOfBiasOnModel(self.matWBias)
		if a != b:
			print "Error #2: your bias model \"", name, "\" has", a, "layer does not match with current bias model", b
			exit()

		self.matWeight = weightLoaded
		self.matWBias = wBiasLoaded

	def initMatLayer(self):
		for n in self.NLayerCount:
			temp = []
			bias = []
			biasWeight = []
			for i in range(n):
				temp.append(0)
				if n == 0:
					continue
				bias.append(0)
				biasWeight.append(self.biasInitWeigt)

			self.matWBias.append(np.array(biasWeight))
			self.matWBiasE.append(np.array(bias))
			self.matNeuronL.append(np.array(temp))

		self.matWBias = np.array(self.matWBias)
		self.matWBiasE = np.array(self.matWBiasE)
		self.matNeuronL = np.array(self.matNeuronL)

		self.miniBatchB = np.array(self.matWBiasE)

	def initMatWeights(self):
		for i in range(len(self.NLayerCount) - 1):
			temp_weights = []
			temp_weights_e = []
			for j in range(self.NLayerCount[i]):
				temp_w = []
				temp_w_e = []
				for k in range(self.NLayerCount[i + 1]):
					temp_w.append(random.random())
					temp_w_e.append(0)
				temp_weights.append(temp_w)
				temp_weights_e.append(temp_w_e)
			self.matWeight.append(np.array(temp_weights))
			self.matWeightE.append(np.array(temp_weights_e))

		self.matWeight = np.array(self.matWeight)
		self.matWeightE = np.array(self.matWeightE)

		self.miniBatchW = np.array(self.matWeightE)

	# init weight by random number
	def initWeights(self):
		for i in range(self.SumNData[len(self.NLayerCount) - 1]):
			self.Weight.append(random.random())
			self.WeightE.append(0)

	def updateWeightAndBias(self):
		self.matWeight -= self.miniBatchW
		self.matWBias -= self.miniBatchB

	# Error adjusting
	def backpropagation(self, expectation):
		deltaRight = (self.matNeuronL[len(self.matNeuronL) - 1] - expectation) * self.sigmoid(self.matNeuronL[len(self.matNeuronL) - 1], True)
		for p in reversed(range(len(self.matNeuronL) - 1)):
			last_w = self.Momentum * self.matWeightE[p]
			self.matWeightE[p] = self.LearningRte * (np.dot(deltaRight.T, self.matNeuronL[p]).T + last_w) + last_w
			self.matWBiasE[p + 1] = self.LearningRte * deltaRight
			temp = np.dot(self.matWeight[p], deltaRight.T)
			deltaRight = temp.T * self.sigmoid(self.matNeuronL[p], True)

		self.miniBatchW += self.matWeightE
		self.miniBatchB += self.matWBiasE

	# feedforward
	def forward(self):
		for p in range(len(self.matNeuronL) - 1):
			self.matNeuronL[p + 1] = self.sigmoid(np.dot(self.matNeuronL[p], self.matWeight[p]) + self.matWBiasE[p + 1])

	def forwardTest(self):
		self.forward()
		return self.matNeuronL[len(self.matNeuronL) - 1]

	def predict(self, input_data):
		self.matNeuronL[0] = np.array([input_data])
		return self.forwardTest()[0]

	def fullTest(self):
		numberOfNodeOutput = self.NLayerCount[-1:][0]
		cumm_error 	= np.array([0.0] * numberOfNodeOutput)
		error_list  = []
		if self.printTestingDetail: print "-----------------------"
		if self.printTestingDetail: print "Full test begin"
		for td in self.TrainData:
			expectation = td[1]
			self.matNeuronL[0] = np.array([td[0]])
			prediction = self.forwardTest()[0]
			if self.printTestingDetail: print "case: ", td[0]
			if self.printTestingDetail: print "prediction: ", prediction
			if self.printTestingDetail: print "expectation: ", expectation
			error_list.append(np.absolute(np.array(prediction) - np.array(expectation)))

		for error_data in error_list: 
			cumm_error += error_data
		cumm_error /= float(len(error_list))
		if self.printTestingDetail: print "End"
		if self.printTestingDetail: print "-----------------------"
		if self.printErrorProgress: print "Average output error: ", cumm_error
		if self.printErrorProgress: print "Cummulative error: ", np.mean(cumm_error)
		if self.printErrorProgress: print "-----------------------"

	def train(self, N):
		for i in range(N):
			if self.printBatchProgress: print "\nTraining batch #" + str(i + 1) + "/" + str(N) + "...."
			countBatch = 0
			for td in self.TrainData:
				if countBatch == 0:
					self.miniBatchW[:] = 0
					self.miniBatchB[:] = 0
				countBatch += 1
				self.matNeuronL[0] = np.array([td[0]])
				self.forward()
				self.backpropagation(np.array([td[1]]))
				if countBatch == self.miniBatchN:
					self.updateWeightAndBias()
					countBatch = 0
			if countBatch != 0: self.updateWeightAndBias()
			if self.testingPeriode > 0 and i % self.testingPeriode == 0: self.fullTest()

	def test(self, data):
		self.matNeuronL[0] = np.array([data])
		return self.forwardTest()
