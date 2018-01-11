#!/usr/bin/env python

import numpy as np
from operations import operations

class ga_main(operations):
    def __init__(self, l_gen, n_pop, n_parents, pb_mut, pb_crs, calc_type,
                 crs_ratio = None, mut_ratio = None):
        super(ga_main, self).__init__(l_gen, n_parents, calc_type, crs_ratio , mut_ratio)
        
        # ---------- function set type
        self.calc_type = calc_type
        # ---------- change available
        self.n_pop = n_pop
        self.pb_mut = pb_mut
        self.pb_crs = pb_crs
        
        # ---------- set parameter
        self.inds = np.empty([n_pop,l_gen])
        self.fitness = None
        self.best_ind_list = np.empty([0,l_gen])
        self.best_fit_list = np.empty([0,l_gen])
        
        # ---------- available parameters
        self._valid_params = ["l_gen","n_pop","n_parents","pb_mut",
                             "pb_crs","crs_ratio", "mut_ratio"]       
    
    def set_params(self, **params):
        if not params:
            return self
        for name, value in params.items():
            name,_,_ = name.partition('__')
            if name not in self._valid_params:
                raise ValueError("{} is not exist in input parameters .".format(name))
            elif "ratio" in name:
                self.set_ratios(**params)
            else:
                setattr(self,name,value)            
        return self   
    
    def get_params(self):
        params = {}
        for name in self._valid_params:
            if getattr(self,name) != None:
                if "func" in name:
                    params[name] = getattr(self,name).__name__
                else :
                    params[name] = getattr(self,name)
        return params
    
    def show_params(self):
        print("----- Parameters ------")
        params = self.get_params()
        for param in self._valid_params:
            try:
                print(str(param) + " = "+ str(params[param]))
            except:
                print(str(param) + " = null")
                
    def make_init_generation(self, n_1 = None):
        if n_1 != None and n_1 != "random" and type(n_1) != int:
            raise TypeError("n_1 should be 'random' or integer type.")
        if self.calc_type == "permutation":
            if n_1 != None:
                raise ValueError("function type is permutation. n_1 is needless.")
            for i in range(self.n_pop):
                self.inds[i] = np.random.permutation(np.arange(self.l_gen))
        
        elif self.calc_type in ["binary","b+p" ]:
            if n_1 == None:
                raise ValueError("Set number of 1.")
            # binary 
            if n_1 == "random" :
                if self.calc_type == "b+p":
                    raise ValueError("When binary + permutation type , n_1 should be integer.")
                for i in range(self.n_pop) :
                    self.inds[i] = np.random.choice([0,1], self.l_gen)
            # permutation of binary type of gen 
            elif n_1 > self.l_gen:
                raise ValueError("The number of 1 is larger than l_gen.")
            elif n_1 == 0 or n_1 == self.l_gen :
                raise ValueError("There are not 0 or 1 in gene.")
            else :
                base_array = np.array([int(0) for i in range(self.l_gen - n_1)]+[int(1) for j in range(n_1)])
                for i in range(self.n_pop) :
                    self.inds[i] = np.random.permutation(base_array)
                    
        self.inds = self.inds.astype(int)
        self.init_ind = self.inds
    
    def get_best_individuals(self):
        self.best_ind = self.inds[np.argmax(self.fitness)]
        self.best_fit = max(self.fitness)
        self.best_ind_list = np.append(self.best_ind_list,self.best_ind.reshape(1,-1),axis=0)
        self.best_fit_list = np.append(self.best_fit_list,self.best_fit)
        
    
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
    
    def calc_onemax_fitness(self):
        fitness = [np.sum(self.inds[i]) for i in range(self.n_pop)]
        return fitness
    
    def calc_sort_fitness(self):
        fitness = [np.sum(np.where(self.inds[i] == 1)[0]) for i in range(self.n_pop)]
        return fitness
    
