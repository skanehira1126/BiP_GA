#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import numpy as np
import copy

class operations(object):
    def __init__(self, l_gen, n_parents, crs_ratio= None, mut_ratio= None):
        self.l_gen = l_gen
        self.n_parents = n_parents
        self.fitness = None
        
        # change ratio
        if type(crs_ratio) == np.ndarray:
            crs_ratio = list(crs_ratio)
        if type(mut_ratio) == np.ndarray:
            mut_ratio = list(mut_ratio)
        
        # ---------- hidden parameters
        self._valid_params=["l_gen","n_parents","crs_ratio", "mut_ratio"]  
        self._changeable_params = ["n_parents","crs_ratio", "mut_ratio"]
        # ----------- functions
        self.slct_funcs= [self.tournament_selection, self.elete_selection, self.roulette_selection]
        self.funcs = {}
        self.funcs["Selection"] = ["tournament", "elete", "roulette"]
        
        # ------ Operation ratios 
        for ratio in [crs_ratio, mut_ratio]:
            if ratio == None:
                continue
            if type(ratio) != list and ratio != None:
                raise TypeError("Probability should be list or numpy.ndarray.")
            elif ratio != None and sum(ratio) != 1:
                raise ValueError("Sum of probability should be 1.")
        if crs_ratio != None:
            crs_ratio = [float(i) for i in crs_ratio]
        if mut_ratio != None:
            mut_ratio = [float(i) for i in mut_ratio]
        
        
        self.crs_ratio = crs_ratio
        self.mut_ratio = mut_ratio
        self._valid_ratios = ["crs_ratio", "mut_ratio"]
        
    def set_params(self, **params):
        if not params:
            return self
        for name, value in params.items():
            name,_,_ = name.partition('__')
            if name not in self._valid_params:
                raise ValueError("{} is not exist in input parameters .".format(name))
            elif name not in self._changeable_params:
                raise ValueError("{} is not changeable.".format(name))
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
        ratios = {}
        for name in self._valid_ratios:
            ratios[name] = getattr(self,name)
        return ratios
    
    def set_fitness(self, fitness):
        self.fitness = fitness
        
    """Selection"""    
    def tournament_selection(self, t_size, population, p_size= None):
        if p_size == None:
            p_size = self.n_parents
        parents = np.empty([0,self.l_gen])
        while parents.shape[0] < p_size:
            tounament = np.random.choice(range(len(population)),t_size)
            fits = np.array([self.fitness[i] for i in tounament])
            parents = np.append(parents,population[tounament[np.argmax(fits)]].reshape(1,-1)
                                ,axis=0)
        return parents
    
    def roulette_selection(self, population, p_size= None):
        if p_size == None:
            p_size = self.n_parents
        sum_fitness = sum(self.fitness)
        if sum_fitness == None:
            raise ValueError("fitness is wrong")
        probability = [float(i)/sum_fitness for i in self.fitness]
        parents = population[np.random.choice(xrange(len(population)), p_size, p = probability, replace = False)]
        return parents
    
    def elete_selection(self, population, p_size= None):
        if p_size == None:
            p_size = self.n_parents
        indexer = np.array(self.fitness).argsort()
        parents = population[indexer[-p_size:]]
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
    
    def GMutation(self, parent , generation , cycle , extra_mut):
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
        slice_index = np.random.choice(range(1,self.l_gen))
        child1 = np.append(p1[:slice_index], p2[slice_index:])
        child2 = np.append(p2[:slice_index], p1[slice_index:])
        self.child = [child1, child2][np.random.choice([0,1])]
        return self.child

    def tp_crossover(self,parents1,parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        slice_index = sorted(np.random.choice(range(1,self.l_gen),2,replace=False))
        # ----- parent1 + parent2 + parent1
        child1 = np.append(p1[:slice_index[0]], p2[slice_index[0]:slice_index[1]])
        child1 = np.append(child1, p1[slice_index[1]:])
        # ----- parent2 + parent1 + parent2
        child2 = np.append(p2[:slice_index[0]], p1[slice_index[0]:slice_index[1]])
        child2 = np.append(child2, p2[slice_index[1]:])
        self.child = [child1, child2][np.random.choice([0,1])]
        return self.child
    
    def uniform_crossover(self, parents1, parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        parents = np.concatenate([p1.reshape(1,-1),p2.reshape(1,-1)],axis=0)
        child = np.array([])
        for i in xrange(self.l_gen):
            index = np.random.choice([0,1])
            child = np.append(child,parents[index,i]) 
        self.child = child
        return self.child
    
    # -------------- permutation encoding
    def cycle_crossover(self, parents1, parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
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
    
    def op_order_crossover(self, parents1, parents2):
        p1, p2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        a = np.random.randint(0,self.l_gen-1)
        c1 = np.append(p1[:a],np.array([i for i in p2 if i not in p1[:a]]))
        c2 = np.append(p2[:a],np.array([i for i in p1 if i not in p2[:a]]))
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    def order_based_crossover(self, parents1, parents2):
        c1,c2 = copy.deepcopy(parents1) ,copy.deepcopy(parents2)
        index = np.random.choice(range(0,self.l_gen),np.random.choice(range(2,self.l_gen-1)),replace=False)
        chng_index1 = sorted([np.where(parents1 == i)[0][0] for i in parents2[index]])
        chng_index2 = sorted([np.where(parents2 == i)[0][0] for i in parents1[index]])
        c1[chng_index1] , c2[chng_index2]= parents2[index] ,parents1[index]
        self.child = [c1,c2][np.random.choice([0,1])]
        return self.child
    
    def position_based_crossover(self,parents1,parents2):
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
        self.child = [c1,c2][np.random.choice([0,1])]
        self.child = self.child.astype(int)
        return self.child
    
    # -------------- binary + permutation encoding 
    def bp_uniform_crossover(self, parents1, parents2):
        c1, c2 = copy.deepcopy(parents1), copy.deepcopy(parents2)
        # check different value 
        index_12 = np.where(np.abs(parents1 - parents2) == 1)[0]
        if len(index_12) == 0:
            if c1.all() == c2.all():
                self.child = c1
                self.child = self.child.astype(int)
                return self.child
            else :
                raise ValueError("Unexpected error.")
        index_1 = np.where(parents1 == 1)[0]
        index_2 = np.where(parents2 == 1)[0]
        # cross point of parents1 and parents2
        cros_ind_1 = [i for i in index_1 if i in index_12]
        cros_ind_2 = [i for i in index_2 if i in index_12]
        
        # do crossover
        new_index_1 = cros_ind_1
        while new_index_1 == cros_ind_1:
            new_index_1 = sorted(np.random.choice(index_12, len(cros_ind_1), replace=False))
        new_index_2 = [i for i in index_12 if i not in new_index_1]
    
        for i in range(self.l_gen):
            if i in new_index_1:
                c1[i] = 1
            elif i in cros_ind_1:
                c1[i] = 0
            if i in new_index_2:
                c2[i] = 1
            elif i in cros_ind_2:
                c2[i] = 0
        self.child = [c1,c2][np.random.choice([0,1])]
        self.child = self.child.astype(int)
        return self.child
    # -------------- multi binary + permutation encoding 
    def mbp_uniform_crossover(self, parents1, parents2):
        p1, p2 = self.get_binary(parents1), self.get_binary(parents2)
        try:    
            self.gen_type
        except:
            self.gen_type = len(p1)+1
        while True:
            child = {}
            for i in range(1,self.gen_type):
                child[i] = self.bp_uniform_crossover(np.array(p1[i]), np.array(p2[i]))
            if self.check_death(child):
                break
            else :
                continue
        self.child = np.array(self.get_ind(child))
        self.child = self.child.astype(int)
        return self.child
    
    def get_binary(self, individual):
        genes = {}
        for num in range(1,self.gen_type):
            ind = []
            for j in individual:
                if j == num:
                    ind.append(1)
                else:
                    ind.append(0)
            genes[num] = ind
        return genes

    def get_ind(self, genes):
        individual = [0 for i in range(self.l_gen)]
        for i in genes:
            for j in range(self.l_gen):
                if genes[i][j] != 0:
                    individual[j] = i
        return individual
    
    def check_death(self, child):
        if len(child) == 0:
            return False
        for l in range(self.l_gen):
            sum_val = 0
            for ntype in range(1,self.gen_type): 
                sum_val = sum_val + child[ntype][l]
            if sum_val >=2 :
                return False       
        return True  
    
    """Mutation"""
    def substitution_mutation(self, parent): # binary
        parent = copy.deepcopy(parent)
        position = np.random.choice(range(self.l_gen), np.random.choice(range(1,int(self.l_gen/4))), replace=False)
        for i in position:
            parent[i] = np.abs(parent[i]-1)
        self.child = parent
        return self.child
        
    def swap_mutation(self, parent): # permutation
        #print("swap mutation")
        parent = copy.deepcopy(parent)
        a,b = np.random.choice(np.arange(self.l_gen),2,replace=False)
        self.child[a], self.child[b] = parent[b], parent[a]
        return self.child
    
    def inversion_mutation(self, parent): # permutation , binary and b+p
        #print("inversion mutation")
        a,b = np.random.choice(np.arange(self.l_gen),2,replace=False)
        pre_parent = copy.deepcopy(parent)
        if a > b:
            a, b = b,a
        for i in range(self.l_gen):
            if a<=i and i<=b:
                self.child[i] = pre_parent[-(self.l_gen-b)-i+a]
            else:
                self.child[i] = pre_parent[i]
        return self.child
         
    def scramble_mutation(self, parent): # permutation ,binary and b+p
        #print("scramble mutation")
        parent = copy.deepcopy(parent)
        a,b = np.random.choice(np.arange(self.l_gen),2,replace=False)
        if a > b:
            a, b = b,a
        self.child[a:b] = np.random.permutation(parent[a:b])
        return self.child
    
    def translocation_mutation(self, parent): # permutation ,binary and b+p
        #print("translocation mutation")
        a,b = np.random.choice(np.arange(1, self.l_gen-1, 1),2,replace=False)
        pre_parent = parent.copy()
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
    
    def bp_swap_mutation(self, parent):
        child = parent.copy()
        index_0 = []
        index_1 = []
        for i in range(self.l_gen):
            if child[i] == 0:
                index_0.append(i)
            elif child[i] == 1:
                index_1.append(i)
            else :
                raise ValueError("Unexpected error. Gen should be 0 or 1.")
        # swap 0 and 1
        child[np.random.choice(index_1)] = 0
        child[np.random.choice(index_0)] = 1
        self.child = child
        return self.child
                
    

