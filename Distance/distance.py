from numpy import *

#欧式距离
def twoPointDistance(a,b):
    d = sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 )
    return d
print ('a,b 二维距离为：',twoPointDistance((1,1),(2,2)))

#三维空间的欧式距离
def threePointDistance(a,b):
    d = sqrt( (a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2 )
    return d
print ('a,b 三维距离为：',threePointDistance((1,1,1),(2,2,2)))

#多维空间的欧式距离
def distance(a,b):
    sum = 0
    for i in range(len(a)):
        sum += (a[i]-b[i])**2
    return sqrt(sum)
print ('a,b 多维距离为：',distance((1,1,2,2),(2,2,4,4)))

def moreBZOSdis(a,b):
    sumnum = 0
    for i in range(len(a)):
        # 计算si 分量标准差
        avg = (a[i]-b[i])/2
        si = sqrt( (a[i] - avg) ** 2 + (b[i] - avg) ** 2 )
        sumnum += ((a[i]-b[i])/si ) ** 2
    return sqrt(sumnum)
print ('a,b 标准欧式距离：',moreBZOSdis((1,2,1,2),(3,3,3,4)))


def moreMHDdis(a,b):
    sum = 0
    for i in range(len(a)):
        sum += abs(a[i]-b[i])
    return sum
print ('a,b 多维曼哈顿距离为：', moreMHDdis((1,1,1,1),(2,2,2,2)) )


def moreQBXFdis(a,b):
    maxnum = 0
    for i in range(len(a)):
        if abs(a[i]-b[i]) > maxnum:
            maxnum = abs(a[i]-b[i])
    return maxnum
print( 'a,b多维切比雪夫距离：' , moreQBXFdis((1,1,1,1),(3,4,3,4)))

def twoCos(a,b):
    cos = (a[0]*b[0]+a[1]*b[1]) / (sqrt(a[0]**2 + a[1]**2) * sqrt(b[0]**2 + b[1]**2) )
    return cos
print ('a,b 二维夹角余弦距离：',twoCos((1,1),(2,2)))

def moreCos(a,b):
    sum_fenzi = 0.0
    sum_fenmu_1,sum_fenmu_2 = 0,0
    for i in range(len(a)):
        sum_fenzi += a[i]*b[i]
        sum_fenmu_1 += a[i]**2
        sum_fenmu_2 += b[i]**2
    return sum_fenzi/( sqrt(sum_fenmu_1) * sqrt(sum_fenmu_2) )
print ('a,b 多维夹角余弦距离：',moreCos((1,1,1,1),(2,2,2,2)))




