from mlpmat import MLP

# init MLP instance
mlp = MLP([2, 5, 3, 1], learning_rate = 0.15)

# load saved model
mlp.loadModel("model/xor2531")

# predict using model
print mlp.predict([1, 1])
print mlp.predict([1, 0])
print mlp.predict([0, 1])
print mlp.predict([0, 0])
