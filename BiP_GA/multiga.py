#!/usr/bin/env python
# -*- coding:utf-8 -*-

import ga_main
import numpy as np
import random

class multi_GA(object):
    def __init__(self, code_type, operation = "both"):
        # ----- emigration flag 
        self.__emigration_flag = False
        # ----- coding_type
        self.ope = operation
        if operation not in ["both" , "either"]:
            raise ValueError("operation should be both or either.")
            
        self.name = code_type
        self.code_type = getattr(ga_main, code_type)
        if code_type == "permutation":
            self.__ncrs, self.__nmut = 4,4
        elif code_type == "binary":
            self.__ncrs, self.__nmut = 3,4
        elif code_type == "bip" or code_type == "bip":
            self.__ncrs, self.__nmut = 1,4
        else :
            raise ValueError("code_type sholud be permutation , binary or bip.")
        
        for_show_func = self.code_type(l_gen = 0, n_pop = 0, n_parents = 0,
                                       pb_mut = None, pb_crs = None)
        self.__valid_selection = for_show_func.funcs["Selection"]
        del for_show_func
    
    def __str__(self):
        return "code type : " + self.name + "\n" + "blocks : " + str(self.blocks) 
        
    def blocks_init(self, l_gen, n_pop, n_parents, blocks = 3):
        self.blocks = int(blocks)
        self.__alphabets = [chr(i) for i in range(65, 65+self.blocks)]
        self.l_gen = l_gen
        if type(n_pop) == list:
            if len(n_pop) == 1:
                n_pop = n_pop * self.blocks
            elif len(n_pop) != self.blocks :
                raise ValueError("len(n_pop) sholud be blocks.")
        elif type(n_pop) == int :
            n_pop = [n_pop] * self.blocks
        else :
            raise TypeError("n_pop should be list or int.")
            
        #for emigration check
        self.min_pop = min(n_pop)
        
        if type(n_parents) == list:
            if len(n_parents) == 1:
                n_parents = n_parents * self.blocks
            elif len(n_parents) != self.blocks :
                raise ValueError("len(n_parents) sholud be blocks.")
        elif type(n_parents) == int :
            n_parents = [n_parents] * self.blocks
        else :
            raise TypeError("n_parents should be list or int.")
        
        self.n_pop_params = {}
        self.n_parents_params = {}
        for i in range(self.blocks):
            self.n_pop_params[self.__alphabets[i]] = n_pop[i]
            self.n_parents_params[self.__alphabets[i]] = n_parents[i]
        
            
    def probs_init(self, mul_pb_crs, mul_pb_mut, mul_crs_ratio , mul_mut_ratio , chng_l_gen_flag = False):
        # ---------- check parameters
        if type(mul_pb_mut) == list:
            if len(mul_pb_mut) == 1:
                mul_pb_mut = mul_pb_mut * self.blocks
            elif len(mul_pb_mut) != self.blocks :
                raise ValueError("len(mul_pb_mut) sholud be blocks.")
        else :
            mul_pb_mut = [mul_pb_mut] * self.blocks
        if type(mul_pb_crs) == list:
            if len(mul_pb_crs) == 1:
                mul_pb_crs = mul_pb_crs * self.blocks
            elif len(mul_pb_crs) != self.blocks :
                raise ValueError("len(mul_pb_crs) sholud be blocks.")
        else :
            mul_pb_crs = [mul_pb_crs] * self.blocks
        
        ### mul_crs_ratio and mul_mut_ration sholud be np.ndarray
        try:
            mul_crs_ratio = np.array(mul_crs_ratio)
            mul_mut_ratio = np.array(mul_mut_ratio)
        except:
            raise ValueError("parameter shape is wrong.")
        ### check ratio shape
        if len(mul_crs_ratio.shape) != 1:
            if mul_crs_ratio.shape[0] != self.blocks:
                raise ValueError("len(mul_crs_ratio) sholud be blocks.")
            if mul_crs_ratio.shape[1] != self.__ncrs:
                raise ValueError("The number of crs_ratio is wrong.")
        else :
            mul_crs_ratio = np.array([mul_crs_ratio]*self.blocks)
        if len(mul_mut_ratio.shape) != 1:
            if mul_mut_ratio.shape[0] != self.blocks:
                raise ValueError("len(mul_mut_ratio) sholud be blocks.")
            if mul_mut_ratio.shape[1] != self.__nmut:
                raise ValueError("The number of mut_ratio is wrong.")
        else :
            mul_mut_ratio = np.array([mul_mut_ratio]*self.blocks)
            
        # ---------- initialize parameters
        self.mul_pb_crs = mul_pb_crs
        self.mul_pb_mut = mul_pb_mut
        self.mul_crs_ratio = mul_crs_ratio
        self.mul_mut_ratio = mul_mut_ratio
        
        # ---------- available parameters
        self._valid_params = ["mul_pb_crs","mul_pb_mut"]
        self._changeable_params = ["mul_pb_crs","mul_pb_mut"]
        
        self.pb_crs_params = {}
        self.pb_mut_params = {}
        self.crs_ratio_params = {}
        self.mut_ratio_params = {}
        for i in range(self.blocks):
            if self.ope == "either":
                if self.mul_pb_crs[i] + self.mul_pb_mut[i] > 1:
                    raise ValueError("sum of probability should be least than 1.")
            self.pb_crs_params[self.__alphabets[i]] = self.mul_pb_crs[i]
            self.pb_mut_params[self.__alphabets[i]] = self.mul_pb_mut[i]
            self.crs_ratio_params[self.__alphabets[i]] = self.mul_crs_ratio[i,:]
            self.mut_ratio_params[self.__alphabets[i]] = self.mul_mut_ratio[i,:]

    
    def coding_init(self, select_func, t_size = None):
        self.t_size = t_size 
        # ----- for selection function
        if type(select_func) == list:
            if len(select_func) == 1:
                select_func = select_func *self.blocks
            elif len(select_func) != self.blocks:
                raise ValueError("len(select_func) sholud be blocks.")
        elif type(select_func) == str :
            select_func = [select_func] *self.blocks
         
        self.select_func = {}
        self.tournament_flag = []
        for i in range(self.blocks):
            if select_func[i] == "tournament":
                if t_size == None:
                    raise ValueError("If tournament selection, need to t_size.")
            if select_func[i] not in self.__valid_selection:
                raise ValueError("Selection function should be elete, roulette or tournament.")

            self.select_func[self.__alphabets[i]] = select_func[i]
        
        self.coding = {}
        for block in self.__alphabets:
            self.coding[block] = self.code_type(l_gen = self.l_gen, n_pop = self.n_pop_params[block], 
                                                n_parents = self.n_parents_params[block],
                                                pb_mut = self.pb_mut_params[block],
                                                pb_crs= self.pb_crs_params[block], 
                                                crs_ratio = self.crs_ratio_params[block],
                                                mut_ratio = self.mut_ratio_params[block], 
                                                chng_l_gen_flag = False, print_flag = False) 
        return self.coding
    
    def show_block_state(self, block="all"):
        if block == "all":
            block = self.coding.keys()
        print("code type : " + str(self.name))
        print("gene length : " + str(self.l_gen))
        for i in block:
            print("-------- block : {} ---------".format(i))
            print("Selection function : " + self.select_func[i])
            print("population : " + str(self.coding[i].n_pop))
            print("parents : " + str(self.coding[i].n_parents))
            print("crossover probability : " + str(self.coding[i].pb_crs))
            print("mutation probability : " + str(self.coding[i].pb_mut))
            print("crossover ratio : " + str(self.coding[i].crs_ratio))
            print("mutation ratio : " + str(self.coding[i].mut_ratio))
            
    def set_params(self, block, **params):
        if not params:
            return self
        self.coding[block].set_params(**params)
        return self.coding
            
    
    def make_init_generation(self, **param):
        self.individuals = {}
        for i in self.coding.keys():
            if param == None:
                self.individuals[i] = self.coding[i].make_init_generation()
            else :
                self.individuals[i] = self.coding[i].make_init_generation(**param)
                
    def set_emigration(self, interval, inds, etype= "random"):
        self.__emigration_flag = True
        if inds > self.min_pop:
                raise ValueError("individuals is too large.")
        self._emigration_param = {}
        self._emigration_param["interval"] = interval
        self._emigration_param["individuals"] = inds
        self._emigration_param["type"] = etype
        return self._emigration_param
        
            
    def evolutinal_step(self, converge, eval_func, **params):
        generation = 1
        while True:
            print("-------- generation : {} --------".format(generation))
            if generation == 1:
                for i in self.coding.keys():
                    fitness = eval_func(self.individuals[i], **params)
                    self.coding[i].set_fitness(fitness)
                    self.coding[i].get_best_individuals()
                    print("{} block best fitness: ".format(i)+ str(max(self.coding[i].fitness)))
                generation += 1
                continue
    
            for i in self.coding.keys():
                """Selection"""
                # tournament
                if self.select_func[i] == "tournament":
                    parents = self.coding[i].tournament_selection(t_size=self.t_size, population = self.individuals[i])
                elif self.select_func[i] == "roulette":
                    parents = self.coding[i].roulette_selection(population = self.individuals[i])
                elif self.select_func[i] == "elete":
                    parents = self.coding[i].elete_selection(population = self.individuals[i])
                
                offspring = np.empty([0, self.l_gen]) ### next generation population
                while offspring.shape[0] != self.coding[i].n_pop :
                    if self.ope == "both":
                        p1, p2 = np.random.choice(parents.shape[0],2,replace=False) 
                        """Crossover"""
                        self.coding[i].Crossover(parents[p1], parents[p2])
                        """Mutation"""
                        self.coding[i].Mutation(parent=self.coding[i].child)
                    else :
                        n = np.random.random()
                        if n < self.coding[i].pb_crs:
                            p1, p2 = np.random.choice(parents.shape[0],2,replace=False) 
                            """Crossover"""
                            self.coding[i].Crossover(parents[p1], parents[p2])
                        elif self.coding[i].pb_crs < n and n < (self.coding[i].pb_crs + self.coding[i].pb_mut):
                            p = np.random.choice(parents.shape[0],1)
                            """Mutation"""
                            self.coding[i].Mutation(parent=parents[p].child)
                    #perm.GMutation(parent=perm.child, generation=gen, extra_mut=0.5, cycle=100)
                    offspring = np.append(offspring, self.coding[i].child.reshape(1,-1),axis=0).astype(int)
                self.coding[i].set_individuals(offspring)
                self.individuals[i] = offspring
                
                
                #----- When emigrate, needless to calclate fitness 
                if self.__emigration_flag == True :
                    if generation % self._emigration_param["interval"] == 0:
                        continue
                
                # calculate fitness
                fitness = eval_func(self.individuals[i])
                self.coding[i].set_fitness(fitness)
                self.coding[i].get_best_individuals()
                                
                print("{} block best fitness: ".format(i)+ str(max(self.coding[i].fitness)))
            
            if self.__emigration_flag == True and generation % self._emigration_param["interval"] == 0:
                new_inds = self.emigration(self._emigration_param)
                for i in self.coding.keys():
                    self.coding[i].set_individuals(new_inds[i])
                    self.individuals[i] = new_inds[i]
                    fitness = eval_func(self.individuals[i])
                    self.coding[i].set_fitness(fitness)
                    print("{} block best fitness: ".format(i)+ str(max(self.coding[i].fitness)))
                
            generation += 1
            if generation > converge:
                break
                
                
    def emigration(self, emigration_param):
        n_inds = emigration_param["individuals"]
        if emigration_param["type"] not in ["random", "fitness"] :
            raise ValueError("etype is random only now")
      
        emigration_perm = np.random.permutation(self.coding.keys())
        move_inds = {}
        left_inds = {}
        for i in self.coding.keys():
            if emigration_param["type"] == "random":
                move_inds[i], left_inds[i] = self.random_selection(self.individuals[i], n_inds)
            elif emigration_param["type"] == "fitness" :
                move_inds[i], left_inds[i] = self.fitness_selection(self.individuals[i], n_inds, self.coding[i].fitness)
                
        new_individuals={}
        for i in range(self.blocks):
            new_individuals[emigration_perm[i]] = np.array(left_inds[emigration_perm[i]] + 
                                                               move_inds[emigration_perm[i-1]])
        return new_individuals
    
    def random_selection(self, individuals, n_inds):
        shuffle = random.sample(individuals, len(individuals))
        move_inds = shuffle[:n_inds]
        left_inds = shuffle[n_inds:]
        return move_inds, left_inds
    
    def fitness_selection(self, individuals, n_inds, fitness):
        inds = len(individuals)
        prob = [float(i)/sum(fitness) for i in fitness]
        move = list(np.random.choice(range(inds), size = n_inds, p = prob, replace=False))
        left = []
        for i in range(inds):
            if i not in move:
                left.append(i)
        move_inds = list(individuals[move])
        left_inds = list(individuals[left])
        
        return move_inds, left_inds
                
        
        
        
                 