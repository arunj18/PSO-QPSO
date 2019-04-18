from typing import TypeVar, List
from copy import copy
from math import sqrt, gamma, pi, sin
import random

import numpy as np
import matplotlib.pyplot as plt

from jmetal.component.archive import BoundedArchive
from jmetal.component.evaluator import Evaluator, SequentialEvaluator
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.operator import Mutation
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from jmetal.component.comparator import DominanceComparator
from jmetal.component.quality_indicator import HyperVolume

R = TypeVar('R')

def random_uniform(l,u):
	return max(10e-8, np.random.uniform(l,u))

class MOQPSO(ParticleSwarmOptimization):
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 max_evaluations: int,
                 mutation: Mutation[FloatSolution],
                 leaders: BoundedArchive[FloatSolution],
                 evaluator: Evaluator[FloatSolution] = SequentialEvaluator[FloatSolution](),
                 reference_point = None,
                 levy: int = 0,
                 levy_decay: int = 0):
        """ This class implements the Multi-Objective variant of Quantum Behaved PSO algorithm  as described in
        :param problem: The problem to solve.
        :param swarm_size: Swarm size.
        :param max_evaluations: Maximum number of evaluations.
        :param mutation: Mutation operator.
        :param leaders: Archive for leaders.
        :param evaluator: An evaluator object to evaluate the solutions in the population.
        """
        super(MOQPSO, self).__init__()
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_evaluations = max_evaluations
        self.mutation = mutation
        self.leaders = leaders
        self.evaluator = evaluator
        self.levy = levy
        self.levy_decay = levy_decay
        self.prev_gbest = None
        self.hypervolume_calculator = HyperVolume(reference_point)

        self.evaluations = 0
        self.beta_swarm = 1.2
        self.g = 0.95

        self.dominance_comparator = DominanceComparator()
        self.constrictors = [(problem.upper_bound[i] - problem.lower_bound[i]) / 5000.0 for i in range(problem.number_of_variables)]

        self.prev_hypervolume = 0
        self.current_hv = 0

        self.hv_changes = []

        self.beta = 3/2
        self.sigma = (gamma(1 + self.beta) * sin(pi * self.beta / 2) / (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)


    def init_progress(self) -> None:
        self.evaluations = 0
        self.leaders.compute_density_estimator()

    def update_progress(self) -> None:
        self.evaluations += 1
        self.leaders.compute_density_estimator()

        observable_data = {'evaluations': self.evaluations,
                           'computing time': self.get_current_computing_time(),
                           'population': self.leaders.solution_list,
                           'reference_front': self.problem.reference_front}

        self.observable.notify_all(**observable_data)

    def is_stopping_condition_reached(self) -> bool:
        completion = self.evaluations / float(self.max_evaluations)
        condition1 = self.evaluations >= self.max_evaluations
        tolerance_cond = False
        '''fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = [],[],[]
        for particle in self.swarm:
            x.append(particle.variables[0])
            y.append(particle.variables[1])
            z.append(particle.objectives[0])
        ax.scatter(x, y, z, c='r', marker='o')
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x,y,z = [],[],[]
        for particle in self.swarm:
            x.append(particle.variables[2])
            y.append(particle.variables[3])
            z.append(particle.objectives[1])
        ax.scatter(x, y, z, c='r', marker='o')
        plt.show()
        '''
        self.beta_swarm = 1.2 - (self.evaluations/self.max_evaluations)*(0.7)
        if(self.prev_gbest is not None):
            gbest = self.evaluator.evaluate([self.prev_gbest],self.problem)
            gbest = gbest[0]
            alpha = 1/gbest.number_of_objectives
            old_score = 0
            for i in range(gbest.number_of_objectives):
                old_score+= alpha*gbest.objectives[i]
            gbest = self.select_global_best()
            gbest = self.evaluator.evaluate([gbest],self.problem)
            gbest = gbest[0]
            new_score = 0
            for i in range(gbest.number_of_objectives):
                new_score+= alpha*gbest.objectives[i]
            if np.abs(new_score - old_score) < 10e-6:
                tolerance_cond = True
            
        condition2 = completion > 0.1 and tolerance_cond
        self.prev_hypervolume = self.current_hv
        return condition1 or condition2


    def create_initial_swarm(self) -> List[FloatSolution]:
        #swarm = lorenz_map(self.swarm_size)
        '''
        new_solution = FloatSolution(number_of_variables, number_of_objectives, number_of_constraints,
                                     lower_bound, upper_bound)
        
        
        '''
        swarm = self.problem.create_solution(self.swarm_size)
        return swarm

    def evaluate_swarm(self, swarm: List[FloatSolution]) -> List[FloatSolution]:
        return self.evaluator.evaluate(swarm, self.problem)

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(particle)

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['local_best'] = copy(particle)

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        pass  # Velocity initialized in the constructor

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        pass

    def update_position(self, swarm: List[FloatSolution]) -> None:
        self.best_global = self.select_global_best()
        #print(self.leaders.solution_list)
        #input()
        self.prev_gbest = self.best_global
        self.current_hv = self.hypervolume_calculator.compute(self.leaders.solution_list)
        self.hv_changes.append(self.current_hv)
        # print("Iteration : {} HV: {}".format(self.evaluations, self.current_hv))
        mbest = []
        for i in range(swarm[0].number_of_variables):
            mbest.append([])
        for i in range(self.swarm_size):
            particle = swarm[i]
            for vari in range(swarm[i].number_of_variables):
                mbest[vari].append(swarm[i].variables[vari])
        for i in range(len(mbest)):
            mbest[i] = sum(mbest[i])/self.swarm_size

        for i in range(self.swarm_size):
            particle = swarm[i]
            best_particle = copy(swarm[i].attributes['local_best'])
            #best_global = self.select_global_best()
            #rint(best_global)
            
            for j in range(particle.number_of_variables):
                psi_1 = random_uniform(0,1)
                psi_2 = random_uniform(0,1)
                P = (psi_1*best_particle.variables[j] + psi_2 * self.best_global.variables[j])/(psi_1 + psi_2)
                u = random_uniform(0,1)
        
                #levy part here
                levy_decayed = 1
                if self.levy :
                    l_u = np.random.normal(0,1) * self.sigma
                    l_v = np.random.normal(0,1)
                    step = l_u / abs(l_v) ** (1 / self.beta)
                    stepsize = 0.01 * step * (1/(0.000001 + particle.variables[j] - self.best_global.variables[j]))
                    levy_decayed = stepsize
                    if self.levy_decay:
                        decay = (1 - (self.evaluations/self.max_evaluations)**self.levy_decay) * random_uniform(0,1)
                        levy_decayed *= decay
                    

                if random_uniform(0,1) > 0.5:
                    particle.variables[j] = P - self.beta_swarm*self.constrictors[j]*mbest[j]*np.log(1/u)*levy_decayed
                else:
                    particle.variables[j] = P + self.beta_swarm*self.constrictors[j]*mbest[j]*np.log(1/u)*levy_decayed

                particle.variables[j] = max(self.problem.lower_bound[j],particle.variables[j])
                particle.variables[j] = min(self.problem.upper_bound[j], particle.variables[j])
        

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            if (i % 6) == 0:
                self.mutation.execute(swarm[i])
        

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            self.leaders.add(copy(particle))

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for i in range(self.swarm_size):
            flag = self.dominance_comparator.compare(
                swarm[i],
                swarm[i].attributes['local_best'])
            if flag != 1:
                swarm[i].attributes['local_best'] = copy(swarm[i])

    def get_result(self) -> List[FloatSolution]:
        return self.leaders.solution_list

    def select_global_best(self) -> FloatSolution:
        leaders = self.leaders.solution_list

        if len(leaders) > 2:
            particles = random.sample(leaders, 2)

            if self.leaders.comparator.compare(particles[0], particles[1]) < 1:
                best_global = copy(particles[0])
            else:
                best_global = copy(particles[1])
        else:
            best_global = copy(self.leaders.solution_list[0])

        return best_global
    
    def get_hypervolume_history(self):
        return self.hv_changes