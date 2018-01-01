#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import numpy as np
import copy

class operations(object):
    def __init__(self, l_gene, n_parents, crs_ratio= None, mut_ratio= None, func_set = "permutation"):
        self.l_gene = l_gene
        self.n_parents = n_parents
        # ----------- functions
        self.slct_funcs= [self.tournament_selection, self.elete_selection, self.roulette_selection]
        self.funcs = {}
        self.funcs["Selection"] = ["tournament", "elete", "roulette"]
        if func_set == "permutation":
            self.crs_funcs = [self.cycle_crossover, self.op_order_crossover,
                              self.order_based_crossover, self.position_based_crossover]
            self.mut_funcs = [self.swap_mutation, self.inversion_mutation, self.scramble_mutation,
                         self.translocation_mutation]
            self.funcs["Crossover"] = ["cycle", "op_order", "order_based", "position_based"]
            self.funcs["Mutation"]  = ["swap", "inversion", "scramble", "translocation"]
            
        elif func_set == "binary":
            self.crs_funcs = [self.op_crossover, self.tp_crossover, self.uniform_crossover]
            self.mut_funcs = [self.substitution_mutation, self.inversion_mutation, self.scramble_mutation,
                         self.translocation_mutation]
            self.funcs["Crossover"] = ["op", "tp", "uniform"]
            self.funcs["Mutation"]  = ["substitution", "inversion", "scramble", "translocation"]
        else :
            raise ValueError("func_set should be permutation or binary.")
        
        print("------- Information of Genetic Algorithm operation  -------")
        for func in self.funcs:
            print(func + ": [" + ", ".join(self.funcs[func]) + " ]")
        
        for ratio in [crs_ratio, mut_ratio]:
            if type(ratio) != list and type(ratio) != np.ndarray and ratio != None:
                raise TypeError("Probability should be list or numpy.ndarray")
            if ratio != None and sum(ratio) != 1:
                raise ValueError("Sum ob probability should be 1.")
        if crs_ratio != None:
            self.crs_ratio = [float(i) for i in crs_ratio]
        if mut_ratio != None:
            self.mut_ratio = [float(i) for i in mut_ratio]
        
        # ------ Operation ratios 
        self.crs_ratio = crs_ratio
        self.mut_ratio = mut_ratio
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
                raise ValueError("The number of {} parameters is wrong.".format(self._valid_ratios[0]))
            if name == self._valid_ratios[1] and len(self.mut_funcs) != len(ratio):
                raise ValueError("The number of {} parameters is wrong.".format(self._valid_ratios[1]))
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
        parents = np.empty([0,self.l_gene])
        while parents.shape[0] < p_size:
            tounament = np.random.choice(range(len(population)),t_size)
            fits = np.array([fitness[i] for i in tounament])
            parents = np.append(parents,population[tounament[np.argmax(fits)]].reshape(1,-1)
                                ,axis=0)
        return parents
    
    def roulette_selection(self, p_size, fitness, population):
        sum_fitness = sum(fitness)
        probability = [i/sum(fitness) for i in fitness]
        parents = population[np.random.choice(xrange(len(population)), p_size,
                                              p = probability, replace = False)]
        return parents
    
    def elete_selection(self, e_size, fitness, population):
        indexer = np.array(fitness).argsort()
        parents = population[indexer[-e_size:]]
        return parents
    
    # ---------- main operation
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

    """Crossover"""
    # ----------- binary encoding
    def op_crossover(self,parents1,parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        slice_index = np.random.choice(range(1,self.l_gene))
        child1 = np.append(p1[:slice_index], p2[slice_index:])
        child2 = np.append(p2[:slice_index], p1[slice_index:])
        self.child = [child1, child2][np.random.choice([0,1])]
        return self.child

    def tp_crossover(self,parents1,parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        slice_index = sorted(np.random.choice(range(1,self.l_gene),2,replace=False))
        # ----- parent1 + parent2 + parent1
        child1 = np.append(p1[:slice_index[0]], p2[slice_index[0]:slice_index[1]])
        child1 = np.append(child1, p1[slice_index[1]:])
        # ----- parent2 + parent1 + parent2
        child2 = np.append(p2[:slice_index[0]], p1[slice_index[0]:slice_index[1]])
        child2 = np.append(child2, p2[slice_index[1]:])
        self.child = [child1, child2][np.random.choice([0,1])]
        return self.child
    
    def uniform_crossover(parents1, parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        parents = np.concatenate([p1.reshape(1,-1),p2.reshape(1,-1)],axis=0)
        child = np.array([])
        for i in xrange(self.l_gene):
            index = np.random.choice([0,1])
            child = np.append(child,parents[index,i]) 
        slef.child = child
        return self.child
    
    # -------------- permutation encoding
    def cycle_crossover(self, parents1, parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        p_list = np.arange(self.l_gene)
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

        for index in np.arange(self.l_gene):
            if index in indexes:
                c1 = np.append(c1,int(p1[index]))
                c2 = np.append(c2,int(p2[index]))
            else :
                c1 = np.append(c1,int(p2[index]))
                c2 = np.append(c2,int(p1[index]))
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    def op_order_crossover(self, parents1, parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        a = np.random.randint(0,self.l_gene-1)
        c1 = np.append(p1[:a],np.array([i for i in p2 if i not in p1[:a]]))
        c2 = np.append(p2[:a],np.array([i for i in p1 if i not in p2[:a]]))
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    def order_based_crossover(self, parents1, parents2):
        c1,c2 = copy.deepcopy(parents1) ,copy.deepcopy(parents2)
        index = np.random.choice(range(0,self.l_gene),np.random.choice(range(2,self.l_gene-1)),replace=False)
        chng_index1 = sorted([np.where(parents1 == i)[0][0] for i in parents2[index]])
        chng_index2 = sorted([np.where(parents2 == i)[0][0] for i in parents1[index]])
        c1[chng_index1] , c2[chng_index2]= parents2[index] ,parents1[index]
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    def position_based_crossover(parents1,parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        chng_index = sorted(np.random.choice(range(10),np.random.choice(range(1,9)),replace=False))
        c1_changed = p1[chng_index]
        c2_changed = p2[chng_index]
        c1_left = np.delete(p1,[int(np.where(p1 == i)[0]) for i in c2_changed])
        c2_left = np.delete(p2,[int(np.where(p2 == i)[0]) for i in c1_changed])
        c1, c2 = np.array([]), np.array([])
        for i in range(10):
            if i in chng_index:
                c1 = np.append(c1, c2_changed[0])
                c2 = np.append(c2, c1_changed[0])
                c1_changed = np.delete(c1_changed, 0)
                c2_changed = np.delete(c2_changed, 0)
            else :
                c1 = np.append(c1, c1_left[0])
                c2 = np.append(c2, c1_left[0])
                c1_left = np.delete(c1_left, 0)
                c2_left = np.delete(c2_left, 0)
        self.child = [c1,c2][np.random.choice([c1,c2])]
        self.child = self.child.astype(int)
        return self.child
    
    """Mutation"""
    def substitution_mutation(self, parent): # binary
        parent = copy.deepcopy(parent)
        position = np.random.choice(range(self.l_gene), np.random.choice(range(1,self.l_gene/4)), replace=False)
        for i in position:
            parent[i] = np.abs(parent[i]-1)
        self.child = parent
        return self.child
        
    def swap_mutation(self, parent): # permutation
        #print("swap mutation")
        parent = copy.deepcopy(parent)
        a,b = np.random.choice(np.arange(self.l_gene),2,replace=False)
        self.child[a], self.child[b] = parent[b], parent[a]
        return self.child
    
    def inversion_mutation(self, parent): # permutation and binary
        #print("inversion mutation")
        a,b = np.random.choice(np.arange(self.l_gene),2,replace=False)
        pre_parent = copy.deepcopy(parent)
        if a > b:
            a, b = b,a
        for i in range(self.l_gene):
            if a<=i and i<=b:
                self.child[i] = pre_parent[-(self.l_gene-b)-i+a]
            else:
                self.child[i] = pre_parent[i]
        return self.child
         
    def scramble_mutation(self, parent): # permutation and binary
        #print("scramble mutation")
        parent = copy.deepcopy(parent)
        a,b = np.random.choice(np.arange(self.l_gene),2,replace=False)
        if a > b:
            a, b = b,a
        self.child[a:b] = np.random.permutation(parent[a:b])
        return self.child
    
    def translocation_mutation(self, parent): # permutation and binary
        #print("translocation mutation")
        a,b = np.random.choice(np.arange(1, self.l_gene-1, 1),2,replace=False)
        pre_parent = parent.copy()
        if a > b:
            a, b = b, a
        n = np.random.choice([-1,1])
        if n == 1 :
            if self.l_gene - b == 1:
                stride = 1
            else :
                stride = np.random.choice(range(1,self.l_gene-b))
        elif n == -1 :
            if a == 1:
                stride = 1
            else :
                stride = np.random.choice(range(1,a))  
        count = 0
        for i in range(self.l_gene):
            if a+stride*n <= i and i <= b+stride*n :
                self.child[i] = pre_parent[i-stride*n]
            elif b+stride*n < i and i <= b and n == -1:
                count = count +1
                self.child[i] = pre_parent[a-n*(count-stride)-1]
            
            elif a <= i and i < a+stride*n and n == 1:
                count = count +1
                self.child[i] = pre_parent[b+n*count]

            else:
                self.child[i] = pre_parent[i]
        return self.child
    

