import numpy as np


def P_generator(MatingPool, Boundary, Coding, MaxOffspring, op_index):

    Num_Op = max(Boundary[0]) + 1  # kexuan number of operations，12

    N = len(MatingPool)
    if MaxOffspring < 1 or MaxOffspring > N:
        MaxOffspring = N
    if Coding == "Real":
        N, D = MatingPool.shape
        ProC = 1
        ProM = 1 / D
        DisC = 20
        DisM = 20
        Offspring = np.zeros((N, D))
        for i in range(0, N, 2):
            beta = np.zeros((D,))
            miu = np.random.random((D,))
            beta[miu <= 0.5] = (2 * miu[miu <= 0.5]) ** (1 / (DisC + 1))
            beta[miu > 0.5] = (2 - 2 * miu[miu > 0.5]) ** (-1 / (DisC + 1))
            beta = beta * ((-1) ** (np.random.randint(0, 2, (D,))))
            beta[np.random.random((D,)) > ProC] = 1

            Offspring[i, :] = ((MatingPool[i, :] + MatingPool[i + 1, :]) / 2) + (
                np.multiply(beta, (MatingPool[i, :] - MatingPool[i + 1, :]) / 2))
            Offspring[i + 1, :] = ((MatingPool[i, :] + MatingPool[i + 1, :]) / 2) - (
                np.multiply(beta, (MatingPool[i, :] - MatingPool[i + 1, :]) / 2))
        Offspring_temp = Offspring[:MaxOffspring, :]
        Offspring = Offspring_temp

        if MaxOffspring == 1:
            MaxValue = Boundary[0, :]
            MinValue = Boundary[1, :]
        else:
            MaxValue = np.tile(Boundary[0, :], (MaxOffspring, 1))
            MinValue = np.tile(Boundary[1, :], (MaxOffspring, 1))

        k = np.random.random((MaxOffspring, D))
        miu = np.random.random((MaxOffspring, D))
        Temp = np.bitwise_and(k <= ProM, miu < 0.5)

        Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]),
                                                        ((2 * miu[Temp] + np.multiply(
                                                            1 - 2 * miu[Temp],
                                                            (1 - (Offspring[Temp] - MinValue[Temp]) / (
                                                                        MaxValue[Temp] - MinValue[Temp])) ** (
                                                                        DisM + 1))) ** (1 / (
                                                                DisM + 1)) - 1))

        Temp = np.bitwise_and(k <= ProM, miu >= 0.5)

        Offspring[Temp] = Offspring[Temp] + np.multiply((MaxValue[Temp] - MinValue[Temp]),
                                                        (1 - ((2 * (1 - miu[Temp])) + np.multiply(
                                                            2 * (miu[Temp] - 0.5),
                                                            (1 - (MaxValue[Temp] - Offspring[Temp]) / (
                                                                        MaxValue[Temp] - MinValue[Temp])) ** (
                                                                        DisM + 1))) ** (1 / (
                                                                DisM + 1))))

        Offspring[Offspring > MaxValue] = MaxValue[Offspring > MaxValue]
        Offspring[Offspring < MinValue] = MinValue[Offspring < MinValue]

    elif Coding == "Binary":

        Offspring = []

        cross_ratio = 0.1

        for i in range(0, N, 2):
            P1 = MatingPool[i].dec.copy()
            P2 = MatingPool[i + 1].dec.copy()

            cross_flag = np.random.rand(1) < cross_ratio

            for j in range(2):
                p1 = np.array(P1[j]).copy()
                p2 = np.array(P2[j]).copy()
                print("**********************开始交叉**********************")
                # ----------------------------crossover-------------------------------
                L1, L2 = len(p1), len(p2)
                L_flag = L1 > L2
                large_L = L1 if L_flag else L2
                common_L = L2 if L_flag else L1
                cross_L = np.random.choice(common_L)

                if cross_flag:
                    p1[:cross_L], p2[:cross_L] = p2[:cross_L], p1[:cross_L]
                print("**********************结束交叉**********************")
                print("**********************开始变异**********************")
                muta_indicator_1, muta_indicator_2 = mutation_indicator(p1.copy(), p2.copy(), op_index)

                muta_p1 = mutation(p1.copy(), op_index, int(Num_Op))
                muta_p2 = mutation(p2.copy(), op_index, int(Num_Op))
                print("**********************结束变异**********************")
                if not L_flag:
                    p1[muta_indicator_1] = muta_p1[muta_indicator_1]
                    p2[muta_indicator_2] = muta_p2[muta_indicator_2]
                else:
                    p1[muta_indicator_2] = muta_p1[muta_indicator_2]
                    p2[muta_indicator_1] = muta_p2[muta_indicator_1]

                p1 = Bound2_least_node(p1, op_index)
                p2 = Bound2_least_node(p2, op_index)

                p = np.random.random()
                if p < 0.5:
                    p1, p2 = start_add_mutation(p1, p2, op_index, int(Num_Op))
                P1[j] = list(p1.copy())
                P2[j] = list(p2.copy())

            # ----------------------------crossover between cell-------------------------------
            if not cross_flag:
                temp_p1 = P1.copy()
                P1[1] = P2[1]
                P2[1] = temp_p1[1]

            Offspring.append(P1)
            Offspring.append(P2)

    return Offspring[:MaxOffspring]


def start_add_mutation(solution_1, solution_2, op_index, Num_op):
    solution_1 = mutation_add(solution_1, op_index, Num_op)
    solution_2 = mutation_add(solution_2, op_index, Num_op)
    return solution_1, solution_2


def mutation_add(solution, op_index, Num_op):
    sign = 0
    op_index_solution = [i for i in op_index if i < len(solution)]
    if len(op_index_solution) < 12:
        select_index = op_index.index(op_index_solution[-1])
        print(select_index)
        op_add = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        op_num = op_add[select_index]
        op_node = np.random.randint(0, 2, op_num)
        op_node[-1] = np.random.choice(Num_op)
        solution = np.concatenate((solution, op_node))
    else:
        return solution
    return solution


def mutation_indicator(solution_1, solution_2, op_index):
    op_index_1 = [i for i in op_index if
                  i < len(solution_1)]
    op_index_2 = [i for i in op_index if i < len(solution_2)]

    mutation_indicator_1 = np.random.rand(len(solution_1), ) < 3 / (
                len(solution_1) - len(op_index_1))
    mutation_indicator_2 = np.random.rand(len(solution_2), ) < 3 / (len(solution_2) - len(op_index_2))

    mutation_indicator_1[op_index_1] = np.random.rand(len(op_index_1), ) < 1 / len(op_index_1)
    mutation_indicator_2[op_index_2] = np.random.rand(len(op_index_2), ) < 1 / len(op_index_2)

    if len(mutation_indicator_1) <= len(mutation_indicator_2):
        return mutation_indicator_1, mutation_indicator_2
    else:
        return mutation_indicator_2, mutation_indicator_1


def mutation(solution, op_index, Num_Op):
    solution_index = [i for i in op_index if i < len(solution)]
    op_candidate = []
    for j, index in enumerate(solution_index):
        A = np.random.choice(Num_Op)
        while A == solution[index]:
            A = np.random.choice(Num_Op)

        op_candidate.append(A)
    op_candidate = np.array(op_candidate)

    zero_index = solution == 0
    one_index = solution == 1
    solution[zero_index] = 1
    solution[one_index] = 0
    solution[solution_index] = op_candidate

    return solution


def Bound2_least_node(solution, op_index):
    solution_index = [i for i in op_index if i < len(solution)]
    Length_before = len(solution_index)
    print(Length_before)
    L = 0
    j = 0
    zero_index = []
    while L < len(solution):
        S = L
        L += 3 + j
        node_j_A = np.array(solution[S:L]).copy()
        node_j = node_j_A[:-1]
        if node_j.sum() - node_j[zero_index].sum() == 0:
            zero_index.extend([j + 2])
        j += 1
    print(zero_index)
    length_now = Length_before - len(zero_index)
    if length_now < 5:
        L = 0
        j = 0
        zero_index = []
        while L < len(solution):
            S = L
            L += 3 + j
            node_j_A = np.array(solution[S:L]).copy()
            node_j = node_j_A[:-1]
            if node_j.sum() - node_j[zero_index].sum() == 0:
                zero_index.extend([j + 2])
                solution[S:L][1] = 1

            j += 1

    return solution


solution = np.array([0, 1, 11, 0, 0, 1, 10, 1, 1, 0, 1, 7])
op = [2, 6, 11, 17, 24, 32, 41, 51, 62, 74, 87, 101]
y = mutation_add(solution, op, 12)
print(y)

print(np.random.random(1))










