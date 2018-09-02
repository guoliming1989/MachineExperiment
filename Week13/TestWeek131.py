import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Week13.testCases_v2 import *
from Week13.planar_utils import sigmoid,plot_decision_boundary,load_planar_dataset,load_extra_datasets

#简单的逻辑回归模型分类
np.random.seed(1)   # 设定一个种子，保证结果的一致性
X,Y = load_planar_dataset()
shape_x = X.shape
shape_y = Y.shape
m = X.shape[1]     #训练集大小
print(shape_x,shape_y,m)

# 2、简单的逻辑回归
# 训练逻辑回归分类器
clf = sklearn.linear_model.LogisticRegressionCV()  #创建分类器对象
clf.fit(X.T,Y.T)                                   #用训练数据拟合分类器模型

# 画逻辑回归的分界线
plot_decision_boundary(lambda x: clf.predict(x), X, Y)   #用训练好的分类器去预测新数据
plt.title("Logistic Regression")
plt.show()

# 预测正确率，数据集不是线性可分的，所以逻辑回归表现不好
LR_predictions = clf.predict(X.T)
print(Y.shape,LR_predictions.shape)
LR_accuracy = (np.dot(Y,LR_predictions)+np.dot(1-Y,1-LR_predictions))/Y.size*100
print ('Accuracy of logistic regression: %d' % LR_accuracy +'%')



