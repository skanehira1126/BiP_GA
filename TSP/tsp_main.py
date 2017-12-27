#!/usr/bin/env python

import numpy as np
from operations import operations

class tsp_ga(operations):
    def __init__(self, l_gen, n_pop, n_parents, e_size, pb_mut, pb_crs, 
                 crs_ratio = None, mut_ratio = None):
        super(tsp_ga, self).__init__(l_gen, n_parents, crs_ratio , mut_ratio)
        
        # ---------- change available
        self.n_pop = n_pop
        self.e_size = e_size
        self.pb_mut = pb_mut
        self.pb_crs = pb_crs
        
        # ---------- set parameter
        self.inds = np.empty([n_pop,l_gen])
        self.fitness = None
        self.best_ind_list = np.empty([0,l_gen])
        self.best_fit_list = np.empty([0,l_gen])
        
        # ---------- calc fitness function
        self.fitness_func = self.calc_dist_fitness
        
        # ---------- available parameters
        self.valid_params = ["l_gen","n_pop","n_parents","e_size","pb_mut",
                             "pb_crs","crs_ratio", "mut_ratio","fitness_func"]       
    
    def set_params(self, **params):
        if not params:
            return self
        for name, value in params.items():
            name,_,_ = name.partition('__')
            if name not in self.valid_params:
                raise ValueError("Input parameter {} is not exist.".format(name))
            setattr(self,name,value)            
        return self   
    
    def get_params(self):
        params = {}
        for name in self.valid_params:
            if getattr(self,name) != None:
                if "func" in name:
                    params[name] = getattr(self,name).__name__
                else :
                    params[name] = getattr(self,name)
        return params
    
    def make_init_generation(self, data_type, n_1 = None):
        if data_type == "permutation":
            print("Making permutaion individuals")
            for i in range(self.n_pop):
                self.inds[i] = np.random.permutation(np.arange(self.l_gen))
        
        elif data_type == "binary":
            if n_1 == None:
                raise VarueError("Set number of 1.")
            elif n_1 > self.l_gen:
                raise ValueError("The number of 1 is larger than l_gen.")
            elif n_1 == 0 or n_1 == self.l_gen :
                raise ValueError("There are not 0 or 1 in gen.")
            else :
                base_array = np.array([int(0) for i in range(self.l_gen - n_1)]+[int(1) for j in range(n_1)])
                for i in range(self.n_pop) :
                    self.inds[i] = np.random.permutation(base_array)
                    
        self.inds = self.inds.astype(int)
        self.init_ind = self.inds
    
    def get_best_individuals(self):
        self.best_ind = self.inds[np.argmax(self.fitness)]
        self.best_ind_list = np.append(self.best_ind_list,self.best_ind.reshape(1,-1),axis=0)
        self.best_fit = max(self.fitness)
        self.best_fit_list = np.append(self.best_fit_list,max(self.fitness))
        
    
    def calc_distance(self,target): 
        """In tsp case , fitness is distance"""
        distance = []
        for i in range(self.n_pop):
            """Start(0,0) , Goal(0,0)"""
            dist = np.sqrt(np.sum((target[self.inds[i,0],:]**2))) + np.sqrt(np.sum((target[self.inds[i,-1],:]**2)))
            for j in range(self.l_gen-1):
                dist = dist + np.sqrt(np.sum((target[self.inds[i,j],:]-target[self.inds[i,j+1],:])**2))
            distance.append(dist)
        return distance
    
    def calc_dist_fitness(self, target):
        fitness = [1./i for i in self.calc_distance(target)]
        return fitness
    
    def set_fitness_func(self, func):
        self.fitness_func = func
    
    def get_fitness(self, **params):
        #print("Fitness calculated by ", self.fitness_func.__name__)
        self.fitness = self.fitness_func(**params)
        return self.fitness