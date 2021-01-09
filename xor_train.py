from mlpmat import MLP

# init MLP instance
mlp = MLP([2, 5, 3, 1], learning_rate = 0.15, print_error_progress = True)

# feed training data
mlp.TrainData.append([[1, 1], [0]])
mlp.TrainData.append([[1, 0], [1]])
mlp.TrainData.append([[0, 1], [1]])
mlp.TrainData.append([[0, 0], [0]])

# training using 10000 epoch
mlp.train(10000)

# test trained mlp
print mlp.predict([1, 1])
print mlp.predict([1, 0])
print mlp.predict([0, 1])
print mlp.predict([0, 0])

# save model
mlp.saveModel("model/xormodel")
