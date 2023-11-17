import numpy as np
from EMO_public import F_distance
from EMO_public import NDsort

def F_EnvironmentSelect(Population,FunctionValue,N):
    FrontValue, MaxFront = NDsort.NDSort(FunctionValue, N)

    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)


    Next = np.zeros((1, N), dtype="int64")



    NoN = np.sum(FrontValue<MaxFront)
    Next[0, :NoN] = np.where(FrontValue <MaxFront)[1]
    Last = np.where(FrontValue==MaxFront)[1]
    Rank =np.argsort(-(CrowdDistance[0,Last]))
    Next[0, NoN:] = Last[Rank[:N-NoN]]


    FrontValue_temp =np.array( [FrontValue[0,Next[0,:]]])
    CrowdDistance_temp = np.array( [CrowdDistance[0, Next[0,:]]])
    FunctionValue_temp = FunctionValue[Next[0,:], :]

    select_index = Next[0,:]


    emili_index_temp = np.array([i for i in range(len(Population)) if i not in select_index])


    emili_pop_temp = [Population[i] for i in emili_index_temp]

    emili_functionvalue_temp = FunctionValue[emili_index_temp,:]


    Population_temp = [Population[i] for i in Next[0,:]]

    return Population_temp, FunctionValue_temp, FrontValue_temp, CrowdDistance_temp, select_index, emili_index_temp, emili_pop_temp,emili_functionvalue_temp
