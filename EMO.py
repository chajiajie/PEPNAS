import torch
import random
import torch.backends.cudnn as cudnn
import numpy as np
import collections, argparse,time,logging,sys
import matplotlib.pyplot as plt

import utils
from EMO_public import P_generator,  NDsort,F_distance,F_mating,F_EnvironmentSelect,elimi_NDsort

from model_training import solution_evaluation
from utils import dagnode,create__dir,Plot_network
from Node import Operations_11_name, NetworkCIFAR

from Build_Dataset import build_search_cifar10, build_search_Optimizer_Loss, build_search_cifar100




class individual():
    def __init__(self, dec):
        self.dec = dec
        self.parameter = []
        self.re_duplicate()

        self.trans2dag()
    def re_duplicate(self):

        for i,cell_dag in enumerate(self.dec):
            L = 0
            j = 0
            zero_index = []
            temp_dec = []
            while L <len(cell_dag):
                S = L
                L +=3+j
                node_j_A = np.array(cell_dag[S:L]).copy()#复制
                node_j = node_j_A[:-1]
                if node_j.sum()- node_j[zero_index].sum()==0:
                    zero_index.extend([j+2])
                else:
                    temp_dec.extend(np.delete(node_j_A, zero_index))
                j+=1

            self.dec[i] = temp_dec.copy()

    def trans2dag(self):
        self.dag = []
        self.num_node = []

        for i in range(2):
            dag = collections.defaultdict(list)
            dag[-1] = dagnode(-1, [], None)
            dag[0] = dagnode(0, [0], None)


            j = 0
            L = 0
            while L < len(self.dec[i]):
                S = L
                L += 3+j
                node_j = self.dec[i][S:L]
                dag[j+1] = dagnode(j+1,node_j[:-1],node_j[-1])
                j+=1
            self.num_node.extend([j])
            self.dag.append(dag)
            del dag

    def evaluate(self,train_queue, valid_queue,args,cur_gen,flag=False):

        pre_epoch = 0
        self.fitness = np.random.rand(4,)
        if flag:
            pre_epoch = 1
        else:
            pre_epoch = args.search_epochs + cur_gen
            if pre_epoch > 10:
                pre_epoch = 10
        search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * pre_epoch
        model = NetworkCIFAR(args, 100, args.search_layers, args.search_channels, self.dag, args.search_use_aux_head,
                         args.search_keep_prob,search_steps,args.search_drop_path_keep_prob,args.search_channels_double)
        if self.parameter != []:
            model.load_state_dict(self.parameter)

        self.fitness[0],self.fitness[1],self.fitness[2],self.fitness[3],self.parameter = solution_evaluation(model,train_queue,valid_queue,args,pre_epoch,flag=flag)

        del  model

class EMO():
    def __init__(self,  args, visualization = False):
        self.args = args
        self.popsize = args.popsize
        self.Max_Gen = args.Max_Gen
        self.Gen = 0
        self.initial_range_node = args.range_node
        self.save_dir =args.save

        self.get_op_index()
        self. op_num = len(Operations_11_name)
        self.max_length = self.op_index[-1]+1
        self.coding = 'Binary'

        self.visualization = visualization

        self.Population = []
        self.Pop_fitness=[]
        self.eliminate_pop = []
        self.elimi_functionvalue = []
        self.finess_best = 0

        self.offspring = []
        self.off_fitness=[]

        self.tour_index = []
        self.FrontValue = []
        self.CrowdDistance =[]
        self.select_index = []

        self.build_dataset()

        self.threshold = 0.08#0.08


    def get_op_index(self):
        self.op_index = []
        L = 0
        for i in range(self.initial_range_node[1]):
            L += 3+i
            self.op_index.extend([L-1])
    def build_dataset(self):
        train_queue, valid_queue = build_search_cifar100(args=self.args, ratio=0.9,num_workers=self.args.search_num_work)
        self.train_queue = train_queue
        self.valid_queue = valid_queue
    def initialization(self):
        for i in range(self.popsize):
            rate = (i+1)/self.popsize # used for controlling the network structure between 'line' and 'Inception'
            node_ = np.random.randint(self.initial_range_node[0],self.initial_range_node[1]+1, 2)#随机返回两个整数，array,(5,13,2)

            list_individual = []
            for i,num in enumerate(node_):
                op = np.random.randint(0, self.op_num, num)
                if i==0:
                    op_c = np.random.randint(0,4,num)
                else:
                    op_c = np.random.randint(4, 10, num)
                in_dicator = np.random.rand(num, ) < 0.8
                op[in_dicator] = op_c[in_dicator]

                L = 2
                dag_list =[]
                for j in range(num):
                    L += 1
                    link = np.random.rand(L-1)
                    link[-1] = link[-1] > rate
                    link[0:2] = link[0:2] < rate
                    link[2:-1] = link[2:-1] < 2 / len(link[2:-1]) if len(link[2:-1]) != 0 else []

                    if link.sum()==0:
                        if rate<0.5:
                            link[-1] = 1
                        else:
                            if np.random.rand(1)<0.5:
                                link[1] = 1
                            else:
                                link[0] = 1

                    link = np.int64(link)
                    link = link.tolist()
                    link.extend([op[j]])
                    dag_list.extend(link)
                list_individual.append(dag_list)

            self.Population.append(individual(list_individual))


        Up_boundary = np.ones((self.max_length))
        Up_boundary[self.op_index] = self.op_num-1
        """Up_boundary = [ 1.  1. 11.  1.  1.  1. 11.  1.  1.  1.  1. 11.  1.  1.  1.  1.  1. 11.
                           1.  1.  1.  1.  1.  1. 11.  1.  1.  1.  1.  1.  1.  1. 11.  1.  1.  1.
                           1.  1.  1.  1.  1. 11.  1.  1.  1.  1.  1.  1.  1.  1.  1. 11.  1.  1.
                           1.  1.  1.  1.  1.  1.  1.  1. 11.  1.  1.  1.  1.  1.  1.  1.  1.  1.
                           1.  1. 11.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. 11.  1.  1.
                           1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1. 11.]"""
        Low_boundary = np.zeros((self.max_length))

        self.Boundary = np.vstack((Up_boundary, Low_boundary))
        self.Pop_fitness = self.evaluation(self.Population)

        self.finess_best = np.min(self.Pop_fitness[:, 0])
        self.save('initial')

    def save(self,path=None):

        if path is None:
            path = 'Gene_{}'.format(self.Gen+1)
        whole_path ='{}/{}/'.format(self.save_dir,path)
        create__dir(whole_path)

        fitness_file = whole_path+'fitness.txt'
        np.savetxt(fitness_file, self.Pop_fitness,delimiter=' ')

        emli_fitness_file = whole_path+'e_fitness.txt'
        np.savetxt(emli_fitness_file, self.elimi_functionvalue,delimiter=' ')
	

        Pop_file = whole_path+'Population.txt'
        with open(Pop_file, "w") as file:
            for j,solution in enumerate(self.Population):
                file.write('solution {}: {} \n'.format(j, solution.dec))


        best_index = np.argmin(self.Pop_fitness[:,0])
        solution = self.Population[best_index]
        Plot_network(solution.dag[0], '{}/{}_conv_dag.png'.format(whole_path, best_index))
        Plot_network(solution.dag[1], '{}/{}_reduc_dag.png'.format(whole_path, best_index))



    def evaluation(self, Pop,flag=False):


        fitness = np.zeros((len(Pop),4))
        for i,solution in enumerate(Pop):
            logging.info('\nsolution: {0:>2d}'.format(i+1))
            print('\nsolution: {0:>2d}'.format(i+1))
            solution.evaluate(self.train_queue,self.valid_queue,self.args,self.Gen,flag=flag)
            fitness[i] = solution.fitness



        return fitness[:,:2]

    def Binary_Envirmental_tour_selection(self):
        self.MatingPool,self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue, self.CrowdDistance)


    def genetic_operation(self):
        offspring_dec= P_generator.P_generator(self.MatingPool, self.Boundary, self.coding, self.popsize,self.op_index)
        offspring_dec = self.deduplication(offspring_dec)
        self.offspring=[individual(i) for i in offspring_dec]
        self.off_fitness = self.evaluation(self.offspring)

    def first_selection(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)
        Population_temp = []
        for i, solution in enumerate(Population):
            Population_temp.append(solution)


        FunctionValue = np.zeros((len(Population_temp),2))
        for i, solution in enumerate(Population_temp):
            FunctionValue[i] = solution.fitness[:2]

        return Population_temp,FunctionValue

    def choose_elimi_pop(self,Population,FunctionValue):
        FunctionValue[:,[0,1]] = FunctionValue[:,[1,0]]
        Population_temp=[]
        size = 0
        # 计算参数平均值
        func_temp = np.sum(FunctionValue, axis=0) / len(FunctionValue)
        FunctionValue_temp = np.array([[1.0, 2.0]])
        for indi in FunctionValue:
            if (indi[1] > func_temp[1]):
                Population_temp.append(Population[size])
                if size == 0:
                    FunctionValue_temp = [indi]
                else:
                    FunctionValue_temp = np.append(FunctionValue_temp, [indi], axis=0)
                size += 1

        return Population_temp,FunctionValue_temp



    def Envirment_Selection(self):


        Population, FunctionValue = self.first_selection()


        Population, FunctionValue, FrontValue, CrowdDistance,select_index,emili_index_temp,emili_pop_temp,emili_functionvalue_temp = F_EnvironmentSelect.\
            F_EnvironmentSelect(Population, FunctionValue, self.popsize)


        if self.eliminate_pop!=[]:
            emili_all_pop_temp = np.concatenate((self.eliminate_pop,emili_pop_temp))
            emili_all_functionvalue_temp = np.concatenate((self.elimi_functionvalue,emili_functionvalue_temp))
        else:
            emili_all_pop_temp = emili_pop_temp
            emili_all_functionvalue_temp = emili_functionvalue_temp

        print("emili_all_pop_temp:    ",emili_all_pop_temp)
        print("emili_all_functionvalue_temp:    ",emili_all_functionvalue_temp)


        #选择参数量大于平均值
        self.eliminate_pop,self.elimi_functionvalue = self.choose_elimi_pop(emili_all_pop_temp,emili_all_functionvalue_temp)






        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index


        self.finess_best = np.min(self.Pop_fitness[:, 0])

    def Envirment_Selection_1(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.eliminate_pop)
        Population_temp = []
        for i, solution in enumerate(Population):
            if solution.fitness[0]<self.finess_best + self.threshold:
                Population_temp.append(solution)

        FunctionValue = np.zeros((len(Population_temp), 2))
        for i, solution in enumerate(Population_temp):
            FunctionValue[i] = solution.fitness[:2]

        Population, FunctionValue, FrontValue, CrowdDistance, select_index, emili_index_temp, emili_pop_temp, emili_functionvalue_temp = F_EnvironmentSelect. \
            F_EnvironmentSelect(Population, FunctionValue, self.popsize)

        # 从20个被淘汰的个体里选10个
        self.eliminate_pop, self.elimi_functionvalue = emili_pop_temp,emili_functionvalue_temp

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index

        self.finess_best = np.min(self.Pop_fitness[:, 0])

    def deduplication(self,offspring_dec):
        pop_dec = [i.dec for i in self.Population]
        dedup_offspring_dec = []
        for i in offspring_dec:
            if i not in dedup_offspring_dec and i not in pop_dec:
                dedup_offspring_dec.append(i)

        return dedup_offspring_dec

    def print_logs(self,since_time=None,initial=False):
        if initial:

            logging.info('********************************************************************Initializing**********************************************')
            print('********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time()-since_time)/60

            logging.info('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                  '*****************************************'.format(self.Gen+1,self.Max_Gen,used_time))

            print('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                  '*****************************************'.format(self.Gen+1,self.Max_Gen,used_time))

    def plot_fitness(self):
        if self.visualization:
            plt.clf()
            plt.scatter(self.Pop_fitness[:, 0], self.Pop_fitness[:, 1])#散点图
            plt.xlabel('Error')
            plt.ylabel('parameters: MB')
            plt.pause(0.001)

    def pre_train(self):
        self.Pop_fitness = self.evaluation(self.Population,flag=True)#True代表上一代种群是否训练1次
        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.popsize)[0]  # 错误率和参数量
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

    def elimi_train(self):
        self.elimi_functionvalue = self.evaluation(self.eliminate_pop,flag=True)
        self.elimi_frontvalue = elimi_NDsort.NDSort(self.elimi_functionvalue,self.popsize//2)[0]
        self.elimi_crowdistance = F_distance.F_distance(self.elimi_functionvalue,self.elimi_frontvalue)
    def Main_loop(self):
        since_time = time.time()
        plt.ion()

        self.print_logs(initial=True)
        self.initialization()
        self.plot_fitness()

        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.popsize)[0]
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

        while self.Gen<self.Max_Gen:
            self.print_logs(since_time= since_time)
            if self.eliminate_pop!=[] and (self.Gen+1)%3==0:
                self.Envirment_Selection_1()
            self.Binary_Envirmental_tour_selection()
            self.genetic_operation()
            print("Pop_fitness:",self.Pop_fitness)
            self.Envirment_Selection()
            self.pre_train()
            self.elimi_train()
            self.plot_fitness()

            self.save()
            self.Gen += 1

        plt.ioff()
        plt.savefig("{}/final.png".format(self.save_dir))

if __name__=="__main__":


    # ===================================  args  ===================================
    # ***************************  common setting******************
    parser = argparse.ArgumentParser(description='test argument')
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('-device', type=str, default='cuda')
    parser.add_argument('-save', type=str, default='result')
    # ***************************  EMO setting******************
    parser.add_argument('-range_node', type=list, default=[5, 12])#[5,12]
    parser.add_argument('-popsize', type=int, default=20)
    parser.add_argument('-Max_Gen', type=int, default=25)

    # ***************************  dataset setting******************
    parser.add_argument('-data', type=str, default="data")
    parser.add_argument('-search_cutout_size', type=int, default=None)  # 16
    parser.add_argument('-search_autoaugment', action='store_true', default=False)
    parser.add_argument('-search_num_work', type=int, default=12, help='the number of the data worker.')

    # ***************************  optimization setting******************
    parser.add_argument('-search_epochs', type=int, default=1)  # 50
    parser.add_argument('-search_lr_max', type=float, default=0.1)  # 0.025 NAO
    parser.add_argument('-search_lr_min', type=float, default=0.001)  # 0 for final training
    parser.add_argument('-search_momentum', type=float, default=0.9)
    parser.add_argument('-search_l2_reg', type=float, default=3e-4)  # 5e-4 for final training
    parser.add_argument('-search_grad_bound', type=float, default=5.0)
    parser.add_argument('-search_train_batch_size', type=int, default=128)
    parser.add_argument('-search_eval_batch_size', type=int, default=500)
    parser.add_argument('-search_steps', type=int, default=50000)
    # ***************************  structure setting******************
    parser.add_argument('-search_use_aux_head', action='store_true', default=True)
    parser.add_argument('-search_auxiliary_weight', type=float, default=0.4)
    parser.add_argument('-search_layers', type=int, default=1)  # 3 for final Network
    parser.add_argument('-search_keep_prob', type=float, default=0.6)  # 0.6 also for final training
    parser.add_argument('-search_drop_path_keep_prob', type=float,
                        default=0.8)  # None 会在训练时提高 精度 和速度, 0.8等 更加耗时但最终训练会提升
    parser.add_argument('-search_channels', type=int, default=16)  # 24:48 for final training
    parser.add_argument('-search_channels_double', action='store_true',
                        default=False)  # False for Cifar, True for ImageNet model
    # parser.add_argument("-add_epochs")

    args = parser.parse_args()
    args.search_steps = int(np.ceil(45000 / args.search_train_batch_size)) * args.search_epochs#np.ceil()计算大于等于值的最小整数
    args.save = '{}/EMO_search_{}'.format(args.save, time.strftime("%Y-%m-%d-%H-%M-%S"))

    create__dir(args.save)

    # ===================================  logging  ===================================
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/logs.log'.format(args.save),
                        level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %I:%M:%S %p')

    logging.info("[Experiments Setting]\n" + "".join(
        ["[{0}]: {1}\n".format(name, value) for name, value in args.__dict__.items()]))

    # ----------------------------------- logging  -------------------------------------

    # ===================================  random seed setting  ===================================
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True

    # -----------------------------------  random seed setting  -----------------------------------

    EMO_NAS = EMO(args,visualization=True)
    EMO_NAS.Main_loop()





