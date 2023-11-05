import numpy as np
from EMO_public import F_distance
from EMO_public import NDsort

def F_EnvironmentSelect(Population,FunctionValue,N):
    FrontValue, MaxFront = NDsort.NDSort(FunctionValue, N)

    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)


    Next = np.zeros((1, N), dtype="int64")
    # np.sum() 求矩阵all 元素和，



    NoN = np.sum(FrontValue<MaxFront)#可以用 int(np.sum(FrontValue<MaxFront,axis=1))代替
    #选择非支配解 放前面
    Next[0, :NoN] = np.where(FrontValue <MaxFront)[1] # 满足条件的 索引 是个列向量，但可以赋值 给行向量,后面的[1]是选择索引的部分
    # 拥挤距离 进行选取
    Last = np.where(FrontValue==MaxFront)[1]
    Rank =np.argsort(-(CrowdDistance[0,Last]))
    #剩余的选择拥挤度大的放在后面
    Next[0, NoN:] = Last[Rank[:N-NoN]]
    # print(np.unique(Next[0,NoN:]))


    # emili_index_temp = np.array()
    # emili_temp = np.array()
    FrontValue_temp =np.array( [FrontValue[0,Next[0,:]]])
    CrowdDistance_temp = np.array( [CrowdDistance[0, Next[0,:]]])
    # Population_temp = Population[Next[0,:],:]
    FunctionValue_temp = FunctionValue[Next[0,:], :]

    select_index = Next[0,:]


    emili_index_temp = np.array([i for i in range(len(Population)) if i not in select_index])#未被选中的索引


    emili_pop_temp = [Population[i] for i in emili_index_temp]    #未被选中的种群

    emili_functionvalue_temp = FunctionValue[emili_index_temp,:]#未被选中的种群的适应度


    Population_temp = [Population[i] for i in Next[0,:]]

    return Population_temp, FunctionValue_temp, FrontValue_temp, CrowdDistance_temp, select_index, emili_index_temp, emili_pop_temp,emili_functionvalue_temp

# p = 10
# front  = np.random.random((p,2))
# population = np.random.random((p,2))
# F_EnvironmentSelect(population,front,3)
# x = np.array([[1,2,3,5,4]])
# rank = np.sum(x<100)
# print(x[0,:3])
#
# x = np.random.random((10,2))
# print(x)
# y = np.where(x<0.5)[1]
# print(y)
# print(len(y))
# sum_temp = np.arange(0,41)
# print(sum_temp)