import numpy as np
from public import sortrows

def F_NDSort(PopObj,Operation):
    list = ["all", "half", "first"]
    kind = list.index(Operation)
    N,M = PopObj.shape
    FrontNO = np.inf*np.ones((1, N))
    MaxFNO = 0
    PopObj, rank = sortrows.sortrows(PopObj)


    while (kind <= 0 and np.sum(FrontNO < np.inf) < N) or (kind <= 1 and np.sum(FrontNO < np.inf) < (N/2)) or(kind <= 2 and MaxFNO < 1):
        MaxFNO += 1
        for i in range(N):
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

    front_temp = np.zeros((1,N))
    front_temp[0, rank] = FrontNO



    return front_temp, MaxFNO





