from abc import ABCMeta, abstractmethod
from typing import Generic, TypeVar, List
from pathlib import Path
import random
import numpy as np
from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution

S = TypeVar('S')


class Problem(Generic[S]):
    """ Class representing problems. """

    __metaclass__ = ABCMeta

    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self, reference_front_path: str):
        self.number_of_variables: int = None
        self.number_of_objectives: int = None
        self.number_of_constraints: int = None

        self.obj_directions: List[int] = []
        self.obj_labels: List[str] = []

        self.reference_front: List[S] = None
        if reference_front_path:
            self.reference_front = self.read_front_from_file_as_solutions(reference_front_path)

    @staticmethod
    def read_front_from_file(file_path: str) -> List[List[float]]:
        """ Reads a front from a file and returns a list.

        :return: List of solution points. """
        front = []
        if Path(file_path).is_file():
            with open(file_path) as file:
                for line in file:
                    vector = [float(x) for x in line.split()]
                    front.append(vector)
        else:
            raise Exception('Reference front file was not found at {}'.format(file_path))

        return front

    @staticmethod
    def read_front_from_file_as_solutions(file_path: str) -> List[S]:
        """ Reads a front from a file and returns a list of solution objects.

        :return: List of solution objects. """
        front = []
        if Path(file_path).is_file():
            with open(file_path) as file:
                for line in file:
                    vector = [float(x) for x in line.split()]
                    solution = FloatSolution(2, 2, 0, [], [])
                    solution.objectives = vector

                    front.append(solution)
        else:
            raise Exception('Reference front file was not found at {}'.format(file_path))

        return front

    @abstractmethod
    def create_solution(self) -> S:
        """ Creates a random solution to the problem.

        :return: Solution. """
        pass

    @abstractmethod
    def evaluate(self, solution: S) -> S:
        """ Evaluate a solution. For any new problem inheriting from :class:`Problem`, this method should be replaced.

        :return: Evaluated solution. """
        pass

    def evaluate_constraints(self, solution: S):
        pass

    def get_name(self) -> str:
        return self.__class__.__name__


class BinaryProblem(Problem[BinarySolution]):
    """ Class representing binary problems. """

    __metaclass__ = ABCMeta

    def __init__(self, rf_path: str = None):
        super(BinaryProblem, self).__init__(reference_front_path=rf_path)

    def create_solution(self) -> BinarySolution:
        pass

    @abstractmethod
    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        pass


class FloatProblem(Problem[FloatSolution]):
    """ Class representing float problems. """

    __metaclass__ = ABCMeta

    def __init__(self, rf_path: str = None):
        super(FloatProblem, self).__init__(reference_front_path=rf_path)
        self.lower_bound = None
        self.upper_bound = None

    def create_solution(self,swarm_size = 1) -> FloatSolution:
        if swarm_size > 1:
            return lorenz_map(swarm_size,self.number_of_variables,self.number_of_objectives,self.number_of_constraints,self.lower_bound,self.upper_bound)
        else:
            new_solution = FloatSolution(self.number_of_variables, self.number_of_objectives, self.number_of_constraints,
                                        self.lower_bound, self.upper_bound)
            #### Chaos initialization start
            new_solution.variables = \
            [random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)] #lorenz_map(self.lower_bound,self.upper_bound,self.number_of_variables)#[random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0) for i in range(self.number_of_variables)] 
            return new_solution
            #### Chaos initialization end
        #print(new_solution.variables)
        #print(self.number_of_variables)
        #input()
        

    @abstractmethod
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        pass


class IntegerProblem(Problem[IntegerSolution]):
    """ Class representing integer problems. """

    __metaclass__ = ABCMeta

    def __init__(self, rf_path: str = None):
        super(IntegerProblem, self).__init__(reference_front_path=rf_path)
        self.lower_bound = None
        self.upper_bound = None

    def create_solution(self) -> IntegerSolution:
        new_solution = IntegerSolution(
            self.number_of_variables,
            self.number_of_objectives,
            self.number_of_constraints,
            self.lower_bound, self.upper_bound)

        new_solution.variables = \
            [int(random.uniform(self.lower_bound[i]*1.0, self.upper_bound[i]*1.0))
             for i in range(self.number_of_variables)]  

        return new_solution

    @abstractmethod
    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        pass

def lorenz(x, y, z):
    s = 10.
    r = 28.
    b = 8/3.0
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def lorenz_map(swarm_size,number_of_variables,number_of_objectives,number_of_constraints,lower_bound,upper_bound):
    chaos_limits = { "x_high" : 22.0 , "x_low" : -22.0 , "y_high" : 30.0 , "y_low" : -30.0 , "z_high" : 55.0 , "z_low" : 0.0 }
    chaos_set = { "xs_list_1" : [] , "ys_list_1" : [] , "zs_list_1" : [] , "xs_list_2" : [] , "ys_list_2" : [] , "zs_list_2" : [] , }
    dt = 0.01
    xs = np.empty((swarm_size+1))
    ys = np.empty((swarm_size+1))
    zs = np.empty((swarm_size+1))
    xs[0], ys[0], zs[0] = (np.random.rand(), np.random.rand(), np.random.rand())
    for i in range(swarm_size):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    chaos_set["xs_list_1"].extend(xs)
    chaos_set["ys_list_1"].extend(ys)
    chaos_set["zs_list_1"].extend(zs)
    xs = np.empty((swarm_size+1))
    ys = np.empty((swarm_size+1))
    zs = np.empty((swarm_size+1))
    xs[0], ys[0], zs[0] = (np.random.rand(), np.random.rand(), np.random.rand())
    for i in range(swarm_size):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    chaos_set["xs_list_2"].extend(xs)
    chaos_set["ys_list_2"].extend(ys)
    chaos_set["zs_list_2"].extend(zs)
    choices = [ "xs_list_1" , "xs_list_2" , "ys_list_1" , "ys_list_2" , "zs_list_1" , "zs_list_2" ]
    #print(chaos_set)
    random.shuffle(choices)
    result = []
    for i in range(swarm_size):
        temp = []
        for i_1 in range(number_of_variables):
            
            temp.append(((chaos_set[choices[i_1]][i] - chaos_limits[choices[i_1][0]+"_low"])/(chaos_limits[choices[i_1][0]+"_high"] - chaos_limits[choices[i_1][0]+"_low"])) * (upper_bound[i_1] - lower_bound[i_1]) + lower_bound[i_1])
            #print(chaos_set[sel_list[i_1][0]+ "s_list_" + str(sel_list[i_1][1])][i_1])
        new_solution = FloatSolution(number_of_variables, number_of_objectives, number_of_constraints,
                                     lower_bound, upper_bound)
        #print(temp)
        #input()
        new_solution.variables = temp
        result.append(new_solution)
    return result
    