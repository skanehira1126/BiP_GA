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
    
    def make_init_generation(self):
        for i in range(self.n_pop):
            self.inds[i] = np.random.permutation(np.arange(self.l_gen))
        self.inds = self.inds.astype(int)
        self.init_ind = self.inds
    
    def Crossover(self,p1,p2):
        if self.crs_ratio == None:
            raise ValueError("crs_ratio is not set.")
        if np.random.random() <= self.pb_crs:
            crs_func = np.random.choice(self.crs_funcs, p = self.crs_ratio)
            self.child = crs_func(p1,p2)
        else :
            self.child = [p1,p2][np.random.choice([0,1])]
        
    def Mutation(self,parent):
        if self.mut_ratio == None:
            raise ValueError("mut_ratio is not set.")
        if np.random.random() <= self.pb_mut:
            mut_func = np.random.choice(self.mut_funcs, p = self.mut_ratio)
            self.child = mut_func(parent)
        else :
            self.child = parent
    
    def GMutation(self,parent,generation,cycle,extra_mut):
        if self.mut_ratio == None:
            raise ValueError("mut_ratio is not set.")
        if generation % cycle == 0:
            probability = extra_mut
        else :
            probability = self.pb_mut
        if np.random.random() <= probability:
            mut_func = np.random.choice(self.mut_funcs, p = self.mut_ratio)
            self.child = mut_func(parent)
        else :
            self.child = parent
    
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