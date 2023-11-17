import numpy as np

def F_distance(FunctionValue,FrontValue):
    N, M = FunctionValue.shape
    CrowdDistance = np.zeros((1, N))
    temp=np.unique(FrontValue)
    Fronts = temp[temp != np.inf]


    for f in range(len(Fronts)):
        Front = np.where(FrontValue == Fronts[f] )[1
        Fmax = np.max(FunctionValue[Front, :], axis=0)
        Fmin = np.min(FunctionValue[Front, :], axis=0)
        for i in range(M):
            Rank = FunctionValue[Front, i].argsort()
            CrowdDistance[0, Front[Rank[0]]] = np.inf
            CrowdDistance[0, Front[Rank[-1]]] = np.inf
            for j in range(1,len(Front)-1,1):
                CrowdDistance[0, Front[Rank[j]]] =  CrowdDistance[0, Front[Rank[j]]] + \
                                                         (FunctionValue[Front[Rank[j + 1]], i] - FunctionValue[Front[Rank[j-1]],i])/(Fmax[i]-Fmin[i])
    return CrowdDistance




