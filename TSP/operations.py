#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import numpy as np

class operations(object):
    def __init__(self, l_gen, n_parents, crs_ratio= None, mut_ratio= None):
        self.l_gen = l_gen
        self.n_parents = n_parents
        # ----------- functions
        self.slct_funcs= [self.tournament_selection, self.elete_selection]
        self.crs_funcs = [self.cycle_crossover, self.op_order_crossover]
        self.mut_funcs = [self.swap_mutation, self.inversion_mutation, self.scramble_mutation,
                     self.translocation_mutation]
        self.funcs = {}
        self.funcs["Selection"] = ["Tournament", "Elete"]
        self.funcs["Crossover"] = ["Cycle", "OP order"]
        self.funcs["Mutation"]  = ["Swap", "Inversion", "Scramble", "Translocation"]
        print("------- Information of Genetic Algorithm operation  -------")
        print(self.funcs)
        
        for ratio in [crs_ratio, mut_ratio]:
            if type(ratio) != list and type(ratio) != np.ndarray and ratio != None:
                raise TypeError("Probability should be list or numpy.ndarray")
            if ratio != None and sum(ratio) != 1:
                raise ValueError("Sum ob probability should be 1.")
        if crs_ratio != None:
            self.crs_ratio = [float(i) for i in crs_ratio]
        if mut_ratio != None:
            self.mut_ratio = [float(i) for i in mut_ratio]
        
        self._valid_ratios = ["crs_ratio", "mut_ratio"]
        
    def get_funcs(self):
        return self.funcs
    
    def set_ratios(self, **params):
        if not params:
            return self
        for name, ratio in params.items():
            if type(ratio) != list and type(ratio) != np.ndarray :
                raise TypeError("Probability should be list or numpy.ndarray")
            if name not in self._valid_ratios:
                raise ValueError("Parameter is crs_ratio or mut_ratio only.")
            if sum(ratio) != 1:
                raise ValueError("Sum of probability should be 1.")
            if name == self._valid_ratios[0] and len(self.crs_funcs) != len(ratio):
                raise ValueError("the number of parameters of {} is wrong.".format(self._valid_ratios[0]))
            if name == self._valid_ratios[1] and len(self.mut_funcs) != len(ratio):
                raise ValueError("the number of parameters of {} is wrong.".format(self._valid_ratios[1]))
            # ---------- set probability
            setattr(self,name,[float(i) for i in ratio])
        return self
    
    def get_ratios(self):
        probs = {}
        for name in self._valid_ratios:
            ratios[name] = getattr(self,name)
        return ratios
         
    """Selection"""    
    def tournament_selection(self, t_size, p_size, fitness, population):
        parents = np.empty([0,self.l_gen])
        while parents.shape[0] < p_size:
            tounament = np.random.choice(range(len(population)),t_size)
            fits = np.array([fitness[i] for i in tounament])
            parents = np.append(parents,population[tounament[np.argmax(fits)]].reshape(1,-1)
                                ,axis=0)
        return parents
    
    def elete_selection(self, e_size, fitness, population):
        indexer = np.array(fitness).argsort()
        parents = population[indexer[-e_size:]]
        return parents
    
    """Crossover"""
    def cycle_crossover(self, p1 ,p2):
        #print("cycle crossover")
        p_list = np.arange(self.l_gen)
        cycles = []
        while len(p_list) != 0:
            cycle = []
            position = p_list[0]
            while True :
                cycle.append(position)
                position= int(np.where(p1==p2[position])[0])
                if position in cycle:
                    break
            cycles.append(cycle)
            for drop in cycle:
                p_list = np.delete(p_list,np.where(p_list == drop)[0])   
        n_cycle = len(cycles)
        indexes = []
        if n_cycle != 1:
            while len(indexes) == 1 :
                indexes = random.sample(cycles,np.random.choice(range(1,n_cycle)))
                indexes = sorted([j for i in list(indexes) for j in i])
        else:
            indexes = sorted([j for i in list(cycles) for j in i])

        c1 = np.array([[]],dtype=int)
        c2 = np.array([[]],dtype=int)

        for index in np.arange(self.l_gen):
            if index in indexes:
                c1 = np.append(c1,int(p1[index]))
                c2 = np.append(c2,int(p2[index]))
            else :
                c1 = np.append(c1,int(p2[index]))
                c2 = np.append(c2,int(p1[index]))
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    def op_order_crossover(self, p1, p2):
        #print("OP order crossover")
        a = np.random.randint(0,self.l_gen-1)
        c1 = np.append(p1[:a],np.array([i for i in p2 if i not in p1[:a]]))
        c2 = np.append(p2[:a],np.array([i for i in p1 if i not in p2[:a]]))
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    """Mutation"""
    def swap_mutation(self, child):
        #print("swap mutation")
        a,b = np.random.choice(np.arange(self.l_gen),2,replace=False)
        self.child[a], self.child[b] = child[b], child[a]
        return self.child
    
    def inversion_mutation(self, child):
        #print("inversion mutation")
        a,b = np.random.choice(np.arange(self.l_gen),2,replace=False)
        pre_child = child.copy()
        if a > b:
            a, b = b,a
        for i in range(self.l_gen):
            if a<=i and i<=b:
                self.child[i] = pre_child[-(self.l_gen-b)-i+a]
            else:
                self.child[i] = pre_child[i]
        return self.child
         
    def scramble_mutation(self, child):
        #print("scramble mutation")
        a,b = np.random.choice(np.arange(self.l_gen),2,replace=False)
        if a > b:
            a, b = b,a
        self.child[a:b] = np.random.permutation(child[a:b])
        return self.child
    
    def translocation_mutation(self, child):
        #print("translocation mutation")
        a,b = np.random.choice(np.arange(1, self.l_gen-1, 1),2,replace=False)
        pre_child = child.copy()
        if a > b:
            a, b = b, a
        n = np.random.choice([-1,1])
        if n == 1 :
            if self.l_gen - b == 1:
                stride = 1
            else :
                stride = np.random.choice(range(1,self.l_gen-b))
        elif n == -1 :
            if a == 1:
                stride = 1
            else :
                stride = np.random.choice(range(1,a))  
        count = 0
        for i in range(self.l_gen):
            if a+stride*n <= i and i <= b+stride*n :
                self.child[i] = pre_child[i-stride*n]
            elif b+stride*n < i and i <= b and n == -1:
                count = count +1
                self.child[i] = pre_child[a-n*(count-stride)-1]
            
            elif a <= i and i < a+stride*n and n == 1:
                count = count +1
                self.child[i] = pre_child[b+n*count]

            else:
                self.child[i] = pre_child[i]
        return self.child


