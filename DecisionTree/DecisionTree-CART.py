import numpy as np

def binSplitDataSet(dataSet, feature, value):
    """
    根据特征切分数据集合
    :param dataSet: 数据集合
    :param feature: 带切分的特征
    :param value: 该特征的值
    :return:
        mat0：切分的数据集合0
        mat1：切分的数据集合1
    """
    # np.nonzero(a)，返回数组a中非零元素的索引值数组
    # np.nonzero(dataSet[:, feature] > value)[0]=1，
    # 下面一行代码表示mat0=dataSet[1,:]即第一行所有列
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    # np.nonzero(dataSet[:, feature] <= value)[0]，表示取第一列中小于0.5的数的索引值，
    # 下面代码表示mat0=dataSet[1,:]即第二、三、四行所有列
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1



if __name__ == '__main__':
    testMat = np.mat(np.eye(4))
    mat0, mat1 = binSplitDataSet(testMat, 1, 0.5)
    print('原始集合：\n', testMat)
    print('mat0:\n', mat0)
    print('mat1:\n', mat1)