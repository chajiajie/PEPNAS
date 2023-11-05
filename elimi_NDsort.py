import numpy as np
from EMO_public import sortrows
#非支配排序
def NDSort(PopObj,Remain_Num):

    N,M = PopObj.shape#（20，2）
    FrontNO = np.inf*np.ones((1, N))
    MaxFNO = 0
    PopObj, rank = sortrows.sortrows(PopObj,order="descend")#按列排序，取得错误率从低到高的排序，没有第二个参数，就是第一列,返回重新排序的和索引


    while (np.sum(FrontNO < np.inf) < Remain_Num):
        MaxFNO += 1
        for i in range(N):#N：20
            if FrontNO[0, i] == np.inf:
                Dominated = False
                for j in range(i-1, -1, -1):
                    if FrontNO[0, j] == MaxFNO:
                        m=2
                        while (m <= M) and (PopObj[i, m-1] >= PopObj[j, m-1]):
                            m += 1
                        Dominated = m > M
                        if Dominated or (M == 2):
                            break
                if not Dominated:
                    FrontNO[0,i] = MaxFNO
    # temp=np.loadtxt("temp.txt")
    # print((FrontNO==temp).all())
    front_temp = np.zeros((1,N))
    front_temp[0, rank] = FrontNO
    # FrontNO[0, rank] = FrontNO 不能这么操作，因为 FrontNO值 在 发生改变 ，会影响后续的结果


    return front_temp, MaxFNO
#
# p = np.random.random((40,2))
# pop_size = 20
# print(p)
# front,max = NDSort(p,pop_size)
# print(np.sum(front!=np.inf))
# print(front,max)
# x = np.random.random((1,10))
# print(x)
# print(np.sum(x<0.5))
# FrontNO = np.ones((1, 5))
# a = np.sum(FrontNO < np.inf)
# print(FrontNO < np.inf)
# b = np.array([[1,2,3],[4,5,6],[0,0,0]])
# print(type(sortrows.sortrows(b)))
# front_temp = np.zeros((2,5))
# front_temp[1, [0,1,2,4,3]] = FrontNO
# print(front_temp)
