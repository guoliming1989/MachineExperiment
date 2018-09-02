import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from Week14.testCases_v3 import *
from Week14.dnn_app_utils_v2 import *

plt.rcParams['figure.figsize'] = (5.0,4.0)      #设置 plots 的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

# 1、数据集
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
num_px = train_x_orig.shape[1]
print(train_x_orig.shape, test_x_orig.shape)


# 显示其中一张图片
index = 10
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y[0,index]) +". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")


# 重铺数据，并标准化
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten  = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255.
test_x  = test_x_flatten/255.
print(train_x.shape, test_x.shape)

# 2、两层神经网络
# 输出 w1 w2 b1 b2
def two_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    m = X.shape[1]
    (n_x, n_h, n_y) = layers_dims
    grads = {}
    costs = []

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X,  W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")

        cost = compute_cost(A2, Y)

        # 初始化反向传播
        dA2 = - (np.divide(Y, A2) - np.divide(1-Y, 1-A2))
        #dA2 = np.power(Y-A2,2) #代价函数升高

        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")

        grads["dW1"] = dW1
        grads["dW2"] = dW2
        grads["db1"] = db1
        grads["db2"] = db2

        parameters = update_parameters(parameters, grads, learning_rate)

        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        b2 = parameters["b2"]

        if print_cost and i%100==0:
            costs.append(cost)
            print("cost after iteration {}:{}".format(i, np.squeeze(cost)))

    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations (per 100)')
    plt.ylabel('cost')
    plt.title("learning rate = " + str(learning_rate))
    plt.show()
    return parameters

# 3、L层神经网络
def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)

        if print_cost and i%100==0:
            costs.append(cost)
            print("cost after iteration %i: %f" % (i,cost))

    plt.plot(np.squeeze(costs))
    plt.xlabel('iterations (per 100)')
    plt.ylabel('cost')
    plt.title("learning rate = " + str(learning_rate))
    plt.show()

    return parameters


# 4、运行两层模型
'''
n_x = train_x.shape[0]
n_h = 7
n_y = 1
layers_dims = [n_x, n_h, n_y]

parameters = two_layer_model(train_x, train_y, layers_dims, 
                             learning_rate=0.01, num_iterations=2500, print_cost=True)

predictions_train = predict(train_x, train_y, parameters)
predictions_test  = predict(test_x, test_y, parameters)
'''

# 5、运行L层模型
layers_dims = [train_x.shape[0], 20, 7, 5,  1]
print(layers_dims)

parameters = L_layer_model(train_x, train_y, layers_dims,
                           learning_rate=0.01, num_iterations = 1000, print_cost = True)

predictions_train = predict(train_x, train_y, parameters)
predictions_test  = predict(test_x, test_y, parameters)


# 6、显示一些标记不正确的图像
print_mislabeled_images(classes, test_x, test_y, predictions_test)

"""
# 7、用自己的图像测试
my_image = "my_image.jpg" # change this to the name of your image file 
my_label_y = [1]          # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) +", your L-layer model predicts a \"" \
       + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
"""
