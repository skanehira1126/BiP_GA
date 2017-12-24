#!/usr/bin/env python

import numpy as np
from operations import operations

class tsp_ga(operations):
    def __init__(self, l_gen, n_pop, n_parents, e_size, pb_mut, pb_crs, 
                 crs_prob =None, mut_prob = None):
        super(tsp_ga, self).__init__(l_gen, n_parents, crs_prob , mut_prob)
        # ----------
        self.n_pop = n_pop
        # ---------- change available
        self.e_size = e_size
        self.pb_mut = pb_mut
        self.pb_crs = pb_crs
        
        # ---------- set parameter
        self.inds = np.empty([n_pop,l_gen])
        self.fitness = None
        self.best_ind_list = np.empty([0,l_gen])
        self.best_fit_list = np.empty([0,l_gen])
        
        # ---------- available parameters
        self.valid_params = ["l_gen","n_pop","n_parents","e_size","pb_mut",
                             "pb_crs","crs_prob", "mut_prob"]       
    
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
            params[name] = getattr(self,name)
        return params
    
    def make_init_generation(self):
        for i in range(self.n_pop):
            self.inds[i] = np.random.permutation(np.arange(self.l_gen))
        self.inds = self.inds.astype(int)
        self.init_ind = self.inds
    
    def calc_fitness(self,target): 
        """In tsp case , fitness is distance"""
        self.fitness = []
        for i in range(self.n_pop):
            """Start(0,0) , Goal(0,0)"""
            fit = np.sqrt(np.sum((target[self.inds[i,0],:]**2))) + np.sqrt(np.sum((target[self.inds[i,-1],:]**2)))
            for j in range(self.l_gen-1):
                fit = fit + np.sqrt(np.sum((target[self.inds[i,j],:]-target[self.inds[i,j+1],:])**2))
            self.fitness.append(fit)
    
    def Crossover(self,p1,p2):
        if self.crs_prob == None:
            raise ValueError("Probability of Crossover is not set.")
        crs_func = np.random.choice(self.crs_funcs, p = self.crs_prob)
        self.child = crs_func(p1,p2)
        
    def Mutation(self,child):
        if self.mut_prob == None:
            raise ValueError("Probability of Mutation is not set.")
        mut_func = np.random.choice(self.mut_funcs, p = self.mut_prob)
        self.child = mut_func(child)
    
    def get_best_individuals(self):
        self.best_ind = self.inds[np.argmin(self.fitness)]
        self.best_ind_list = np.append(self.best_ind_list,self.best_ind.reshape(1,-1),axis=0)
        self.best_fit = min(self.fitness)
        self.best_fit_list = np.append(self.best_fit_list,min(self.fitness))


