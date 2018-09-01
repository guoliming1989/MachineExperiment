from math import log, exp

dataSet_neg = [[u'青年', u'否', u'否', u'一般'],
               [u'青年', u'否', u'否', u'好'],
               [u'青年', u'否', u'否', u'一般'],
               [u'中年', u'否', u'否', u'一般'],
               [u'中年', u'否', u'否', u'好'],
               [u'老年', u'否', u'否', u'一般'],
               ]

dataSet_pos = [[u'青年', u'是', u'否', u'好'],
               [u'青年', u'是', u'是', u'一般'],
               [u'中年', u'是', u'是', u'好'],
               [u'中年', u'否', u'是', u'非常好'],
               [u'中年', u'否', u'是', u'非常好'],
               [u'老年', u'否', u'是', u'非常好'],
               [u'老年', u'否', u'是', u'好'],
               [u'老年', u'是', u'否', u'好'],
               [u'老年', u'是', u'否', u'非常好'],
               ]


class LaplaceEstimate(object):
    """
    拉普拉斯平滑处理的贝叶斯估计
    """

    def __init__(self):
        self.d = {}  # [词-词频]的map
        self.total = 0.0
        self.none = 1

    def exists(self, key):
        return key in self.d

    def getsum(self):
        return self.total

    def get(self, key):
        if not self.exists(key):
            return False, self.none
        return True, self.d[key]

    # 计算P(x | y)
    def getprob(self, key):
        return float(self.get(key)[1]) / self.total

    def samples(self):
        """
        获取全部样本
        :return:
        """
        return self.d.keys()

    def add(self, key, value):
        self.total += value
        # map {key:value} ={'好':2}
        if not self.exists(key):
            self.d[key] = 1
            self.total += 1
        self.d[key] += value


class Bayes(object):
    def __init__(self):
        self.d = {}
        self.total = 0

    # 参数计算
    def train(self, data):
        for d in data:
            c = d[1]
            # 对每个分类进行统计 建立map d[pos] 和 d[neg]
            if c not in self.d:
                self.d[c] = LaplaceEstimate()  # 生成拉普拉斯平滑
            # 对特征向量中每个随机变量进行统计
            for word in d[0]:
                self.d[c].add(word, 1)
        self.total = sum(map(lambda x: self.d[x].getsum(), self.d.keys()))

    def classify(self, x):
        tmp = {}
        # 循环每一个分类标签
        for c in self.d:
            tmp[c] = log(self.d[c].getsum()) - log(self.total)
            for word in x:
                tmp[c] += log(self.d[c].getprob(word))
        ret, prob = 0, 0
        for c in self.d:
            now = 0
            try:
                for otherc in self.d:
                    now += exp(tmp[otherc] - tmp[c])
                now = 1 / now
            except OverflowError:
                now = 0
            if now > prob:
                ret, prob = c, now
        return (ret, prob)


class Sentiment(object):
    def __init__(self):
        self.classifier = Bayes()
    def train(self, neg_docs, pos_docs):
        data = []
        # 合并特征向量和分类标签
        for sent in neg_docs:
            data.append([sent, u'neg'])
        for sent in pos_docs:
            data.append([sent, u'pos'])
        self.classifier.train(data)
    def classify(self, sent):
        return self.classifier.classify(sent)


s = Sentiment()
# 测试贷款申请样本数据表
s.train(dataSet_neg, dataSet_pos)
# 测试分类数据的准确率
print("----------neg data------------")
for sent in dataSet_neg:
    print("是否贷款 否",sent,s.classify(sent))
print("----------pos data------------")
for sent in dataSet_pos:
    print("是否贷款：是",sent,s.classify(sent))