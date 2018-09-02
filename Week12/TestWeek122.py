import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import pylab
from PIL import Image
from Week12.lr_utils import load_dataset
from scipy import ndimage

# 1、下载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 2、显示图像
index = 90
#plt.imshow(train_set_x_orig[index])
#pylab.show()   # 加上才能显示图片
# squeeze()函数：除去size=1的维度，（4,1,3）变成（4,3），（4,2,3）则不变
print ("y = " + str(train_set_y[:, index]) + ", it's a '" \
    + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# 3、显示图像大小
m_train = train_set_x_orig.shape[0]
m_test  = test_set_x_orig.shape[0]
num_px  = train_set_x_orig.shape[1]

print("TrainSet:   " + str(train_set_x_orig.shape))
print("TestSet:    " + str(test_set_x_orig.shape))
print("TrainLabel: " + str(train_set_y.shape))
print("TestLabel:  " + str(test_set_y.shape))

# 4、reshape图片,且转换成列向量,三维图像转化为列向量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten  = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
print(train_set_x_flatten.shape,test_set_x_flatten.shape)

# 5、标准化数据集
train_set_x = train_set_x_flatten/255
test_set_x  = test_set_x_flatten/255

# 6、
def sigmoid(z):
    s = 1.0/(1 + np.exp(-z))
    return s

# 7、初始化参数,根据参数初始化输入神经节点的个数
def initialize_zeros(dim):
    w = np.zeros([dim,1])
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b

# 8、forward and backward propagation
# 计算损失函数、梯度
# 8、forward and backward propagation
# 计算损失函数、梯度
def propagate(w,b,x,y):
    '''
    function:
        implement the cost function and gradient
    Arguments:
        w --- 权重 （num_px*num_px*3, 1）
        b --- 偏置
        X --- 输入 （num_px*num_px*3, 样本数）
        Y --- 标签
    return：
        cost --- 逻辑回归的 log 损失函数
        dw   ---
        db   ---
    '''
    m = x.shape[1]
    # 前向传播
    y_   = sigmoid(np.dot(w.T,x) + b)
    cost = -(1.0/m)*np.sum(y*np.log(y_) + (1-y)*np.log(1-y_))
    # 后向传播
    dw = (1.0/m)*np.dot(x,(y_-y).T)
    db = (1.0/m)*np.sum(y_-y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw":dw, "db":db}
    return grads,cost

# w,b,x,y = np.array([[1.],[2.]]), 2, np.array([[1,2,-1],[3,4,-3]]), np.array([[1,0,1]])
# grads, cost = propagate(w,b,x,y)
# print(grads,cost)

# 9、optimization 优化算法(梯度下降)
def optimize(w, b, x, y, num_iterations, learning_rate, print_cost = False):
    '''
    This function optimizes w and b by running a gradient descent algorithm
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    rerutn: params, grads, cost
    '''
    costs = []
    for i in range(num_iterations):
        grads,cost = propagate(w,b,x,y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - dw*learning_rate
        b = b - db*learning_rate
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i%100==0:
            print("cost after iteration %i : %f" %(i,cost))
    params = {"w":w, "b":b}
    grads  = {"dw":dw, "db":db}
    return params,grads,costs

# params, grads, costs = optimize(w, b, x, y, num_iterations= 100, learning_rate = 0.009, print_cost = False)
# print(params,grads)

# 10、预测
def predict(w,b,x):
    m = x.shape[1]
    y_p = np.zeros([1,m])
    w = w.reshape(x.shape[0],1)
    y_ = sigmoid(np.dot(w.T,x)+b)
    for i in range(y_.shape[1]):
        if y_[0,i] > 0.5:
            y_p[0,i]=1
        else:
            y_p[0,i]=0
    assert(y_p.shape == (1,m))
    return y_p

# w = np.array([[0.1124579],[0.23106775]])
# b = -0.3
# x = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
# print ("predictions = " + str(predict(w, b, x)))

# 11、将所有的功能合并到模型中
def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w,b = initialize_zeros(x_train.shape[0])
    parameters,grads,costs = optimize(w,b,x_train,y_train,num_iterations,learning_rate,print_cost)
    w = parameters["w"]
    b = parameters["b"]
    y_p_train = predict(w,b,x_train)
    y_p_test  = predict(w,b,x_test)

    print("train accuracy:{} %".format(100-np.mean(np.abs(y_p_train-y_train))*100))
    print("test accuracy:{} % ".format(100-np.mean(np.abs(y_p_test-y_test))*100))
    d = {"costs":costs,
         "y_p_train":y_p_train,
         "y_p_test":y_p_test,
         "w":w,
         "b":b,
         "learning_rate":learning_rate,
         "num_iterations":num_iterations}
    return d

# 训练集正确率99%，测试集正确率70%，过拟合了
d = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations=2000,learning_rate=0.005,print_cost=True)


# 12、画出代价函数和梯度
'''
costs = np.squeeze(d["costs"]) 
plt.plot(costs)
plt.ylabel("cost")
plt.xlabel("iterations(per hundreds)")
plt.title("learning rate = "+str(d["learning_rate"]))
plt.show()
'''
# 13、学习率的选择
learning_rate = [0.01,0.001,0.0001]
models = {}
for i in learning_rate:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,
                           num_iterations=1500,learning_rate=i,print_cost=False)

for i in learning_rate:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
plt.ylabel("cost")
plt.xlabel("iteration")

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# 14、用自己的图像来测试
fname = 'data/cat_in_iran.jpg'
image = np.array(ndimage.imread(fname,flatten=False))
my_image = scipy.misc.imresize(image,size=(num_px,num_px)).reshape((1,num_px*num_px*3)).T
my_predicted_image = predict(d["w"],d["b"],my_image)

plt.imshow(image)
pylab.show()
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" \
    + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
