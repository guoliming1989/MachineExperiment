import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from Week21.init_utils import *

# matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()

# 1、初始化参数
# 零初始化、随机初始化、其他初始化
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)] = np.zeros((layers_dims[l], layers_dims[l-1]))
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
    return parameters
def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*10
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
    return parameters
def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*np.sqrt(2./layers_dims[l-1])
        parameters['b'+str(l)] = np.zeros((layers_dims[l],1))
    return parameters

# 2、神经网络模型
def model(X, Y, learning_rate=0.01, num_iterations=10000, print_cost=True, initialization="he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]

    # initialize parameters dictionary
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(0,num_iterations):
        a3, cache = forward_propagation(X, parameters)
        cost = compute_loss(a3, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i%500==0:
            print("cost after iteration {}:{}".format(i, cost))
            costs.append(cost)

    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (per 100)")
    plt.title("learning rate = " + str(learning_rate))
    plt.show()
    return parameters

# 3、测试3种初始化
# 模型训练
parameters = model(train_X, train_Y, num_iterations=20000, initialization="zeros")
print('W1 = '+ str(parameters["W1"]))
print('W2 = '+ str(parameters["W2"]))
print('W3 = '+ str(parameters["W3"]))
print('b1 = '+ str(parameters["b1"]))
print('b2 = '+ str(parameters["b2"]))
print('b3 = '+ str(parameters["b3"]))

# 预测训练集、测试集
print("on the train set: ")
predictions_train = predict(train_X, train_Y, parameters)
print("on the test set: ")
predictions_test  = predict(test_X, test_Y, parameters)

print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))

# 画出分界线
plt.title("Model With Zeros Initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)