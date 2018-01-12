#!/usr/bin/env python

import numpy as np
from operations import operations

class permutation_ga(operations):
    def __init__(self, l_gen, n_pop, n_parents, pb_mut, pb_crs,
                 crs_ratio = None, mut_ratio = None):
        super(permutation_ga, self).__init__(l_gen, n_parents, crs_ratio , mut_ratio)
        
        # --------- function set
        self.crs_funcs = [self.cycle_crossover, self.op_order_crossover,
                          self.order_based_crossover, self.position_based_crossover]
        self.mut_funcs = [self.swap_mutation, self.inversion_mutation, self.scramble_mutation,
                         self.translocation_mutation]
        self.funcs["Crossover"] = ["cycle", "op_order", "order_based", "position_based"]
        self.funcs["Mutation"]  = ["swap", "inversion", "scramble", "translocation"]
        
        # ---------- change available
        self.n_pop = n_pop
        self.pb_mut = pb_mut
        self.pb_crs = pb_crs
        
        # ---------- set parameter
        self.inds = np.empty([self.n_pop, self.l_gen])
        self.fitness = None
        self.best_ind_list = np.empty([0, self.l_gen])
        self.best_fit_list = np.empty([0, self.l_gen])
        
        # ---------- available parameters
        self._valid_params.extend(["pb_crs","pb_mut"])
        self._changeable_params.extend(["pb_crs","pb_mut"])
        
        print("------- Information of Genetic Algorithm operation  -------")
        print("calclation type : permutation") 
        for func in self.funcs:
            print(func + ": [" + ", ".join(self.funcs[func]) + " ]")
                
    def make_init_generation(self):
        for i in range(self.n_pop):
            self.inds[i] = np.random.permutation(np.arange(self.l_gen))
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
    
    # ------ for traveling salseman problem : permutation encoding
    def calc_dist_fitness(self, target):
        fitness = [1./i for i in self.calc_distance(target)]
        return fitness

class binary_ga(operations):
    def __init__(self, l_gen, n_pop, n_parents, pb_mut, pb_crs, 
                 crs_ratio = None, mut_ratio = None):
        super(binary_ga, self).__init__(l_gen, n_parents, crs_ratio , mut_ratio)
        
        # ---------- function set type
        self.crs_funcs = [self.op_crossover, self.tp_crossover, self.uniform_crossover]
        self.mut_funcs = [self.substitution_mutation, self.inversion_mutation, self.scramble_mutation,
                         self.translocation_mutation]
        self.funcs["Crossover"] = ["op", "tp", "uniform"]
        self.funcs["Mutation"]  = ["substitution", "inversion", "scramble", "translocation"]

        # ---------- change available
        self.n_pop = n_pop
        self.pb_mut = pb_mut
        self.pb_crs = pb_crs
        
        # ---------- set parameter
        self.inds = np.empty([self.n_pop, self.l_gen])
        self.fitness = None
        self.best_ind_list = np.empty([0,self.l_gen])
        self.best_fit_list = np.empty([0,self.l_gen])
        
        print("------- Information of Genetic Algorithm operation  -------")
        print("calclation type : binary" ) 
        for func in self.funcs:
            print(func + ": [" + ", ".join(self.funcs[func]) + " ]")
        
        # ---------- available parameters
        self._valid_params.extend(["pb_crs","pb_mut"])
        self._changeable_params.extend(["pb_crs","pb_mut"])
                
    def make_init_generation(self, n_1): # binary type
        if n_1 != "random" and type(n_1) != int:
            raise TypeError("n_1 should be 'random' or integer type.")
        if n_1 == "random" :
            for i in range(self.n_pop) :
                self.inds[i] = np.random.choice([0,1], self.l_gen)
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
        
    # ------ for onemax problem : binary encoding
    def calc_onemax_fitness(self):
        fitness = [np.sum(self.inds[i]) for i in range(self.n_pop)]
        return fitness
    

    
class bip_ga(operations):
    def __init__(self, l_gen, n_pop, n_parents, pb_mut, pb_crs, 
                 crs_ratio = [1], mut_ratio = None):
        super(bip_ga, self).__init__(l_gen, n_parents, crs_ratio , mut_ratio)
        
        # ---------- function set type
        self.crs_funcs = [self.bp_uniform_crossover]
        self.mut_funcs = [self.bp_swap_mutation, self.inversion_mutation, self.scramble_mutation, self.translocation_mutation]
        self.funcs["Crossover"] = ["bp_uniform"]
        self.funcs["Mutation"]  = ["bp_swap", "inversion", "scramble", "translocation"]
        
        # ---------- change available
        self.n_pop = n_pop
        self.pb_mut = pb_mut
        self.pb_crs = pb_crs
        
        # ---------- set parameter
        self.inds = np.empty([self.n_pop, self.l_gen])
        self.fitness = None
        self.best_ind_list = np.empty([0, self.l_gen])
        self.best_fit_list = np.empty([0, self.l_gen])
        
        # ---------- available parameters
        self._valid_params.extend(["pb_crs","pb_mut"])
        self._changeable_params.extend(["pb_crs","pb_mut"])
        
        print("------- Information of Genetic Algorithm operation  -------")
        print("calclation type : binary permutation") 
        for func in self.funcs:
            print(func + ": [" + ", ".join(self.funcs[func]) + " ]")
                
    def make_init_generation(self, n_1): # binary type
        if type(n_1) != int:
            raise TypeError("n_1 should be integer type.")
        
        if n_1 > self.l_gen:
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
        
    
    # ------ for 01 sort problem : binary + permutation problem
    def calc_sort_fitness(self):
        fitness = [np.sum(np.where(self.inds[i] == 1)[0]) for i in range(self.n_pop)]
        return fitness    