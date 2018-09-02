import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from Week21.testCases import *
from Week21.reg_utils import *

#matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 1、下载数据
train_X, train_Y, test_X, test_Y = load_2D_dataset()

# 2-1、L2正则化: 计算损失函数
def compute_cost_with_regularization(A3, Y, parameters, lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)
    L2_regularization_cost = (1./m*lambd/2)*(np.sum(np.square(W1))+np.sum(np.square(W2))+np.sum(np.square(W3)))

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

# 2-2、L2正则化：反向传播
def backward_propagation_with_regularization(X, Y, cache, lambd):
    m = X.shape[1]
    (Z1,A1,W1,b1,  Z2,A2,W2,b2,  Z3,A3,W3,b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T) + lambd/m * W3
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T) + lambd/m * W2
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T) + lambd/m * W1
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# 3-1、dropout：前向传播
# 三层网络结构，input---hidden---hidden---output
# dropout只应用到隐含层，不用到输入层和输出层
def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    '''
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    '''
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)

    # 第一隐含层dropout
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1/keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)

    # 第二隐含层dropout
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2/keep_prob

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1,D1,A1,W1,b1,  Z2,D2,A2,W2,b2,  Z3,A3,W3,b3)

    return A3, cache

# 3-2、dropout：反向传播
def backward_propagation_with_dropout(X, Y, cache, keep_prob=0.5):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2
    dA2 = dA2 / keep_prob

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1
    dA1 = dA1 / keep_prob

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

# 4、神经网络模型
def model(X, Y, learning_rate=0.3, num_iterations=10000, print_cost=True, lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 20, 3, 1]
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        # (1)前向传播
        # 是否使用 Dropout
        if keep_prob == 1:
            a3,cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3,cache = forward_propagation_with_dropout(X, parameters, keep_prob)
        # (2)损失函数
        # 是否正则化
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # (3)后向传播
        assert(lambd==0 or keep_prob==1)  # 正则化和 Dropout 不能同时使用

        if lambd==0 and keep_prob==1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # (4)更新参数
        parameters = update_parameters(parameters, grads, learning_rate)

        # (5)每迭代1000步，打印一次损失函数
        if print_cost and i%1000==0:
            costs.append(cost)
            print("cost after iteration {}:{}".format(i, cost))

    # 画出costs
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iterations (per 100)")
    plt.title("learning rate = " + str(learning_rate))
    plt.show()
    return parameters


# 5、训练模型，观察训练集、测试集正确率
# 依次是：无正则化、L2正则化

# 训练模型 =================================================== 选择模式
parameters = model(train_X, train_Y, num_iterations=10000,lambd=0, keep_prob=1)

print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

# 画出分界面
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)