import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s

def load_planar_dataset():
    np.random.seed(1)
    m = 400              # 样本总数
    N = int(m/2)         # 每类样本数
    D = 2                # 维度
    X = np.zeros((m,D))  # 初始化，每行是一个样本
    Y = np.zeros((m,1), dtype='uint8') # 标签 (0 for red, 1 for blue)
    a = 4                # 花的最大长达
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2  # theta
        r = a*np.sin(4*t) + np.random.randn(N)*0.2   # radius
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j
    X = X.T
    Y = Y.T
    return X, Y

def setcolor(x):
    color=[]
    for i in range(x.shape[1]):
        if x[:,i]==1:
            color.append('b')
        else:
            color.append('r')
    return color

def load_extra_datasets():
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)
    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure

def plot_decision_boundary(model, X, y):

    # 设置数据的边界，并向外扩充一点
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # 生成以 h 为步长的格子
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    # ravel()与flatten()都是平铺
    # np.r_按行组合array，np.c_按列组合array
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    # contourf(x,y,f(x,y),alpha=0.75,cmap='jet')
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    color = setcolor(y)
    plt.scatter(X[0, :], X[1, :], c=color, cmap=plt.cm.Spectral)

if __name__=="__main__":
    X,Y = load_planar_dataset()
    print(X.shape,Y.shape)
    color =setcolor(Y)
    # 显示花
    plt.scatter(X[0,:],X[1,:],s=20,c=color,cmap=plt.cm.Spectral)
    plt.show()