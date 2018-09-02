import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io
from Week21.testCases import *
from Week21.gc_utils import *
"""
计算原理：计算一个点左右二边的导数，来确定梯度下降
方法：存储计算的中间值正向传播A和W ，反向传播的值dA和dW，验证梯度下降的值
注意事项：
梯度检验很慢，近似梯度计算成本高。 
所以不能每一步训练都进行梯度检验，只需要检验几次。（用来验证代码的正确性） 
梯度检验不适用于dropout，应该使用完整的网络检验梯度，然后再添加dropout 
梯度检验，计算梯度近似值、反向传播梯度之间的接近程度
"""


# 1-1 、计算一维线性前向传播 J(theta) = theta * x
def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J),(J(theta) = theta * x)
    Arguments:
    x     -- a real-valued input
    theta -- our parameter, a real number as well
    Returns:
    J -- the value of function J, computed using the formula J(theta) = theta * x
    """
    J = theta * x
    return J

# 1-2、计算一维反向传播（导数/梯度） dtheta = J(theat)'
def backward_propagation(x, theta):
    """
    Returns:
    dtheta -- the gradient of the cost with respect to theta
    """
    dtheta = x
    return dtheta

# 1-3、一维梯度检验
def gradient_check(x, theta, epsilon=1e-7):
    """
    Implement the backward propagation

    Arguments:
    x       -- a real-valued input
    theta   -- our parameter, a real number as well
    epsilon -- tiny shift to the input to compute approximated gradient

    Returns:
    difference -- approximated gradient and the backward propagation gradient
    """
    # 计算 gradapprox , epsilon足够小对于limit来说
    theta_plus  = theta + epsilon
    theta_minus = theta - epsilon
    J_plus  = forward_propagation(x, theta_plus)
    J_minus = forward_propagation(x, theta_minus)
    gradapprox = (J_plus - J_minus)/(2 * epsilon)
    # 计算 grad
    grad = backward_propagation(x, theta)
    # 计算 difference
    numerator = np.linalg.norm(grad - gradapprox)                     # 分子
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)   # 分母
    difference = numerator/denominator
    if difference < 1e-7:
        print("the gradient is correct!")
    else:
        print("the gradient is wrong!")
    return difference

# 2-1 计算N维前向传播
def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost) presented in Figure 3.
    Arguments:
    X -- training set for m examples
    Y -- labels for m examples
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (5, 4)
                    b1 -- bias vector of shape (5, 1)
                    W2 -- weight matrix of shape (3, 5)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    Returns:
    cost -- the cost function (logistic cost for one example)
    """
    # retrieve parameters
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache

# 2-2 计算N维反向传播
def backward_propagation_n(X, Y, cache):
    """
    Implement the backward propagation presented in figure 2.

    Arguments:
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    cache -- cache output from forward_propagation_n()

    Returns:
    gradients -- A dictionary with the gradients of the cost with respect to each parameter, activation and pre-activation variables.
    """

    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims = True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims = True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients



# 2-3、N维梯度检验
def gradient_check_n(parameters, gradients, X, Y, epsilon=1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    grad -- output of backward_propagation_n
    x -- input, of shape (input size, 1)
    y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient

    Returns:
    difference --the approximated gradient and the backward propagation gradient
    """
    # 设置变量
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus     = np.zeros((num_parameters,1))
    J_minus    = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))

    # 计算 gradapprox
    for i in range(num_parameters):
        theta_plus  = np.copy(parameters_values)
        theta_plus[i][0]  = theta_plus[i][0] + epsilon
        J_plus[i], _  = forward_propagation_n(X, Y, vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] = theta_minus[i][0] - epsilon
        J_minus[i], _ = forward_propagation_n(X, Y, vector_to_dictionary(theta_minus))

        gradapprox[i] = (J_plus[i] - J_minus[i])/(2. * epsilon)

    numerator   = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference  = numerator/denominator
    if difference > 1e-7:
        print(difference)
        print ("There is a mistake in the backward propagation!")
    else:
        print(difference)
        print ("Your backward propagation works perfectly fine!")
    return difference

X, Y, parameters = gradient_check_n_test_case()
cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)