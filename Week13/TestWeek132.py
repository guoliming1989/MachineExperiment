import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Week13.testCases_v2 import *
from Week13.planar_utils import sigmoid,plot_decision_boundary,load_planar_dataset,load_extra_datasets

# 下载额外的数据集
np.random.seed(1)   # 设定一个种子，保证结果的一致性
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

dataset = "blobs"
X,Y = datasets[dataset]
X,Y = X.T,Y.reshape(1,Y.shape[0])

if dataset == "blobs":
    Y = Y % 2
print(X.shape, Y.shape, X.shape[1])

def setcolor(Y):
    color=[]
    for i in range(Y.shape[1]):
        if Y[:,i]==1:
            color.append('b')
        else:
            color.append('r')
    return color

#显示数据
plt.scatter(X[0,:], X[1:], s=30, c=setcolor(Y), cmap=plt.cm.Spectral)
#plt.show() #加上才显示


# 3、神经网络模型
# 3-1 定义三层网络结构,获取神经元输入与输出节点个数
def layer_sizes(X,Y):
    """
        Arguments:
        X -- input dataset of shape (input size, number of examples)
        Y -- labels of shape (output size, number of examples)
        Returns:
        n_x -- the size of the input layer
        n_h -- the size of the hidden layer
        n_y -- the size of the output layer
        """
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x,n_h,n_y)


# 3-2 初始化模型参数，定义2层神经元节点参数
def initialize_parameters(n_x,n_h,n_y):
    """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)
    print(W1)
    W2 = np.random.randn(n_y,n_h)
    print(W2)
    b1 = np.zeros((n_h,1))
    b2 = np.zeros((n_y,1))
    assert(W1.shape == (n_h,n_x))
    assert(W2.shape == (n_y,n_h))
    assert(b1.shape == (n_h,1))
    assert(b2.shape == (n_y,1))
    parameters = {"W1":W1, "W2":W2, "b1":b1, "b2":b2}
    return parameters

# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# 3-3 计算前向传播,计算A和激励函数的值
def forward_propagation(X,parameters):
    """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)
        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    #assert(A2.shape == (1,X.shape[1]))
    cache = {"Z1":Z1, "Z2":Z2, "A1":A1, "A2":A2}
    return A2,cache

# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)


# 3-4 计算损失函数
def compute_cost(A2,Y,parameters):
    """
        Computes the cross-entropy cost given in equation (13)
        Arguments:
        A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        parameters -- python dictionary containing your parameters W1, b1, W2 and b2
        Returns:
        cost -- cross-entropy cost given equation (13)
    """

    m = Y.shape[1]
    #计算交叉熵损失函数
    logprobs = np.multiply(np.log(A2),Y) + np.multiply(np.log(1-A2),1-Y)
    cost = -np.sum(logprobs)/m
    cost = np.squeeze(cost)
    assert(isinstance(cost,float))
    return cost
# A2, Y_assess, parameters = compute_cost_test_case()
# print("cost = " + str(compute_cost(A2, Y_assess, parameters)))

# 3-5 计算反向传播过程
def backward_propagation(parameters,cache,X,Y):
    """
        Implement the backward propagation using the instructions above.
        Arguments:
        parameters -- python dictionary containing our parameters
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)
        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2-Y
    dW2 = 1/m * np.dot(dZ2,A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = 1/m * np.dot(dZ1,X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1":dW1, "dW2":dW2, "db1":db1, "db2":db2}
    return grads

# parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
# grads = backward_propagation(parameters, cache, X_assess, Y_assess)
# print ("dW1 = "+ str(grads["dW1"]))
# print ("db1 = "+ str(grads["db1"]))
# print ("dW2 = "+ str(grads["dW2"]))
# print ("db2 = "+ str(grads["db2"]))

# 3-6 更新参数
def update_parameters(parameters,grads,learning_rate=1.2):
    """
        Updates parameters using the gradient descent update rule given above
        Arguments:
        parameters -- python dictionary containing your parameters
        grads -- python dictionary containing your gradients
        Returns:
        parameters -- python dictionary containing your updated parameters
    """
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    dW2 = grads["dW2"]
    db1 = grads["db1"]
    db2 = grads["db2"]

    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    b1 -= learning_rate * db1
    b2 -= learning_rate * db2
    parameters = {"W1":W1, "W2":W2, "b1":b1, "b2":b2}
    return parameters

# 3-7 模型
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
       Arguments:
       X -- dataset of shape (2, number of examples)
       Y -- labels of shape (1, number of examples)
       n_h -- size of the hidden layer
       num_iterations -- Number of iterations in gradient descent loop
       print_cost -- if True, print the cost every 1000 iterations
       Returns:
       parameters -- parameters learnt by the model. They can then be used to predict.
    """
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    for i in range(0,num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate=0.3)

        if print_cost and i%100==0:
            print("cost after iteration %i: %f" % (i,cost))
    return parameters

# X_assess, Y_assess = nn_model_test_case()
# parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# 3-8 预测
def predict(parameters, X):
    A2,cache = forward_propagation(X,parameters)
    predictions = np.around(A2)
    return predictions


# 3-9 运行测试代码
# 下载数据，训练模型
parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# 预测正确率
predictions = predict(parameters, X)
print(Y.shape,predictions.shape)
accuracy = float(np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/Y.size*100
print('Accuracy : %d' % accuracy +'%')

#  画边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()