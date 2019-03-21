from typing import TypeVar, List
from copy import copy
from math import sqrt, gamma, pi, sin
import random

import numpy as np

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
                 levy: int = 0,
                 levy_decay: int = 0,
                 reference_point = None):
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

        self.hypervolume_calculator = HyperVolume(reference_point)

        self.evaluations = 0

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
        condition2 = completion > 0.01 and (self.current_hv - self.prev_hypervolume) < 10e-10
        self.prev_hypervolume = self.current_hv
        return condition1 or condition2


    def create_initial_swarm(self) -> List[FloatSolution]:
        #swarm = lorenz_map(self.swarm_size)
        '''
        new_solution = FloatSolution(number_of_variables, number_of_objectives, number_of_constraints,
                                     lower_bound, upper_bound)
        
        
        '''
        #for _ in range(self.swarm_size-1):  #### lorenz gives problems, use tent map instead
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
        best_global = self.select_global_best()
        self.current_hv = self.hypervolume_calculator.compute(self.leaders.solution_list)
        self.hv_changes.append(self.current_hv)
        # print("Iteration : {} HV: {}".format(self.evaluations, self.current_hv))
        for i in range(self.swarm_size):
            particle = swarm[i]
            best_particle = copy(swarm[i].attributes['local_best'])
            best_global = self.select_global_best()
            
            for j in range(particle.number_of_variables):
                psi_1 = random_uniform(0,1)
                psi_2 = random_uniform(0,1)
                P = (psi_1*best_particle.variables[j] + psi_2 * best_global.variables[j])/(psi_1 + psi_2)
                u = random_uniform(0,1)
                L = 1/self.g * np.abs(particle.variables[j] - P)

                #levy part here
                levy_decayed = 1
                if self.levy :
                    l_u = np.random.normal(0,1) * self.sigma
                    l_v = np.random.normal(0,1)
                    step = l_u / abs(l_v) ** (1 / self.beta)
                    stepsize = 0.01 * step * (1/(0.0000001 + particle.variables[j] - best_global.variables[j]))
                    levy_decayed = stepsize
                    if self.levy_decay:
                        decay = (1 - (self.evaluations/self.max_evaluations)**self.levy_decay) * random_uniform(0,1)
                        levy_decayed *= decay
                    

                if random_uniform(0,1) > 0.5:
                    particle.variables[j] = P - self.constrictors[j]*L*np.log(1/u)*levy_decayed
                else:
                    particle.variables[j] = P + self.constrictors[j]*L*np.log(1/u)*levy_decayed

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
    
    def get_hypvervolume_history(self):
        return self.hv_changes