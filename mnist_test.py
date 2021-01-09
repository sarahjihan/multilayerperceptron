from mlpmat import MLP
from image import ImageReader
import numpy as np

# mlp layer: 784 nodes (input) -> 15 nodes (hidden) -> 10 nodes (output)
mlp = MLP([784, 15, 10], learning_rate = 0.001, momentum = 0.05, mini_batch = 100, print_error_progress = True)

def simpulkan(arr):
    # print arr
    maxIdx = 0
    for i in range(len(arr)):
        if arr[i] > arr[maxIdx]:
            maxIdx = i

    return maxIdx

mlp.loadModel("model/mnistall784-15-10Z")

print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/0.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/1.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/2.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/3.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/4.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/5.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/6.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/7.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/8.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/9.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/00.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/11.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/22.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/33.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/44.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/55.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/66.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/77.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/88.png")) / 255))
print simpulkan(mlp.predict(np.array(ImageReader().read("mnist_test/99.png")) / 255))
