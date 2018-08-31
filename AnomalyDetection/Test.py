import numpy as np
import array

# 参数估计函数（就是求均值和方差）
def estimateGaussian(X):
    m, n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    mu = np.mean(X, axis=0)  # axis=0表示列，每列的均值,axis=1表示行的操作
    sigma2 = np.var(X, axis=0)  # 求每列的方差
    return mu, sigma2



if __name__ == "__main__":
    X = np.array([[-9,1], [7,2]])
    print(X)
    a,b =estimateGaussian(X)
    print(a)
    print(b)