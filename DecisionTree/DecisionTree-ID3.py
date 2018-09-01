from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree
from sklearn.externals.six import StringIO  #格式的预处理
"""
树以代表训练样本的单个结点开始（步骤1）。
如果样本都在同一个类，则该结点成为树叶，并用该类标号（步骤2 和3）。
否则，算法使用称为信息增益的基于熵的度量作为启发信息，选择能够最好地将样本分类的属性（步骤6）。该属性成为该结点的“测试”或“判定”属性（步骤7）。在算法的该版本中，
所有的属性都是分类的，即离散值。连续属性必须离散化。
对测试属性的每个已知的值，创建一个分枝，并据此划分样本（步骤8-10）。
算法使用同样的过程，递归地形成每个划分上的样本判定树。一旦一个属性出现在一个结点上，就不必该结点的任何后代上考虑它（步骤13）。
递归划分步骤仅当下列条件之一成立停止：
(a) 给定结点的所有样本属于同一类（步骤2 和3）。
(b) 没有剩余属性可以用来进一步划分样本（步骤4）。在此情况下，使用多数表决（步骤5）。
这涉及将给定的结点转换成树叶，并用样本中的多数所在的类标记它。替换地，可以存放结
点样本的类分布。
(c) 分枝
test_attribute = a i 没有样本（步骤11）。在这种情况下，以 samples 中的多数类
创建一个树叶（步骤12）
"""


#读取文件
allElectronicsData=open(r'data/test.csv','rt')#读取csv文件
reader=csv.reader(allElectronicsData)
headers=next(reader)#定义的第一行
print(headers)

featureList=[]   #数据的输入特性，使用的是数值型的值
labelList=[]
for row in reader:
    labelList.append(row[len(row)-1])#添加标签，取每行的最后一个数字
    #print(labelList)
    rowDict={}#取每行的特征值
    for i in range(1,len(row)-1):
        rowDict[headers[i]]= row[i]#每一行设置字典项
    featureList.append(rowDict)
print(featureList)
vec=DictVectorizer()   #直接在vec对象上调用方法，对于feature值的转化,转化为0或1的属性
dummyX=vec.fit_transform(featureList).toarray()
print("dummyX:"+str(dummyX))
print(vec.get_feature_names())
print("labelList:"+str(labelList))

lb=preprocessing.LabelBinarizer()#对于label的转化
dummyY=lb.fit_transform(labelList)
print("dummyY:"+str(dummyY))

clf=tree.DecisionTreeClassifier(criterion='entropy')  #分类树的参数的选取，基于信息熵来度量标准
clf=clf.fit(dummyX,dummyY)   #填入参数，调用fit，构建决策树
print("clf:"+str(clf))

with open("data/test.dot",'w')as f:
    f=tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=f)
    #将原0、1的数值转化为属性值
# predict
oneRowX = dummyX[0, :]
print("oneRowX:" + str(oneRowX))
# 修改这一行的数据，然后执行
newRowX = oneRowX
newRowX[0] = 1
newRowX[1] = 0
print("newRowX:" + str(newRowX))
# 添加一个中括号
predictedY = clf.predict([newRowX])
# print(help(clf.predict))
print("predictedY:" + str(predictedY))

# 将dot文件转化为pdf文件，dot -Tpdf iris.dot -o outpu.pdf
# 决策树转化
# dot -Tpdf D:\PythonWork\TeachingPython\src\DecsionTree\test.dot -o D:\PythonWork\TeachingPython\src\DecsionTree\outpu.pdf





