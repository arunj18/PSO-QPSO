from jmetal.core.solution import FloatSolution
from jmetal.core.problem import FloatProblem
from penalties import l2_equality_penalty as equality_penalty, l2_inequality_penalty as inequality_penalty, modified_inequality_penalty

def interior_score(radius, density, alpha, beta):
    return (radius**alpha) * (density**beta)

def surface_score(escape_velocity, surface_temperature, gamma, delta):
    return (escape_velocity**gamma) * (surface_temperature**delta)

def ceesa_score(escape_velocity, surface_temperature, radius, density, orb_eccentricity, rho, nu, e, t, v, d, r):
    return (r*(radius**rho) + d*(density**rho) + t*(surface_temperature**rho) + v*(escape_velocity**rho) + e*(orb_eccentricity**rho))**(nu/rho)

def get_drs_objective(radius, density, escape_velocity, surface_temperature):
    global interior_score, surface_score

    tolerance = 10e-7
    error = 10e-6


    def f1(x):
        score = interior_score(radius, density,x[0],x[1])
        objective = -score +  inequality_penalty((x[0] + x[1] - 1), error) + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error)
        return objective

    def f2(x):
        score = surface_score(escape_velocity, surface_temperature,x[0],x[1])
        objective = -score + inequality_penalty((x[0] + x[1] - 1), error) + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error) 
        return objective

    class MultiObjectiveCDH(FloatProblem):
        def __init__(self, rf_path: str = None):
            super(MultiObjectiveCDH, self).__init__(rf_path = rf_path)
            self.number_of_objectives = 2
            self.number_of_variables = 5
            self.number_of_constraints = 2
            self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
            self.obj_labels = ['f(x)', 'f(y)']

            self.lower_bound = [-2.0 for _ in range(self.number_of_variables)]
            self.upper_bound = [2.0 for _ in range(self.number_of_variables)]

            self.lower_bound[-1] = 0.0
            self.upper_bound[-1] = 5.0

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound
        
        def evaluate(self,solution: FloatSolution) -> FloatSolution:
            alpha = solution.variables[0]
            beta = solution.variables[1]
            constant_factor = solution.variables[-1]
            gamma = (constant_factor * alpha * escape_velocity) / radius 
            solution.variables[2] = gamma
            delta = solution.variables[3]

            solution.objectives[0] = f1([alpha, beta])
            solution.objectives[1] = f2([gamma, delta])
            return solution

        def evaluate_constraints(self,solution: FloatSolution) -> None:
            constraints = [0.0 for _ in range(self.number_of_constraints)]

            alpha = solution.variables[0]
            beta = solution.variables[1]
            delta = solution.variables[2]
            gamma = solution.variables[3]

            constraints[0] = (alpha + beta - 1) + error
            constraints[1] = (gamma + delta - 1) + error


            overall_constraint_violation = 0.0
            number_of_violated_constraints = 0.0

            for constrain in constraints:
                if constrain > 0.0:
                    overall_constraint_violation += constrain
                    number_of_violated_constraints += 1

            # print("{} {}".format(overall_constraint_violation, number_of_violated_constraints))
            solution.attributes['overall_constraint_violation'] = overall_constraint_violation
            solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

        # def evaluate_constraints(self,solution: FloatSolution) -> None:
        #     constraints = [0.0 for _ in range(self.number_of_constraints)]

        #     alpha = solution.variables[0]
        #     beta = solution.variables[1]
        #     delta = solution.variables[2]
        #     gamma = solution.variables[3]

        #     constraints[0] = (alpha + beta - 1) - tolerance
        #     constraints[1] = -(alpha + beta - 1) - tolerance
        #     constraints[2] = (delta + gamma -1) - tolerance
        #     constraints[3] = -(delta + gamma -1) - tolerance

        #     overall_constraint_violation = 0.0
        #     number_of_violated_constraints = 0.0

        #     for constrain in constraints:
        #         if constrain > 0.0:
        #             overall_constraint_violation += constrain
        #             number_of_violated_constraints += 1

        #     # print("{} {}".format(overall_constraint_violation, number_of_violated_constraints))
        #     solution.attributes['overall_constraint_violation'] = overall_constraint_violation
        #     solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

        
        def get_name(self):
            return "Multi Objective CDH"

    return MultiObjectiveCDH()

def get_crs_objective(radius, density, escape_velocity, surface_temperature):
    global interior_score, surface_score

    tolerance = 10e-7
    error = 10e-6


    def f1(x):
        score = interior_score(radius, density,x[0],x[1])
        objective = -score + equality_penalty(x[0] + x[1] -1, tolerance)  + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error)
        return objective

    def f2(x):
        score = surface_score(escape_velocity, surface_temperature,x[0],x[1])
        objective = -score + equality_penalty(x[0] + x[1] -1, tolerance) + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error) 
        return objective

    class MultiObjectiveCDH(FloatProblem):
        def __init__(self, rf_path: str = None):
            super(MultiObjectiveCDH, self).__init__(rf_path = rf_path)
            self.number_of_objectives = 2
            self.number_of_variables = 5
            self.number_of_constraints = 4
            self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
            self.obj_labels = ['f(x)', 'f(y)']

            self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
            self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

            self.lower_bound[-1] = 0.0
            self.upper_bound[-1] = 5.0

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound
        
        def evaluate(self,solution: FloatSolution) -> FloatSolution:
            alpha = solution.variables[0]
            beta = solution.variables[1]
            constant_factor = solution.variables[-1]
            gamma = (constant_factor * alpha * escape_velocity) / radius 
            solution.variables[2] = gamma
            delta = solution.variables[3]
            solution.objectives[0] = f1([alpha, beta])
            solution.objectives[1] = f2([gamma, delta])
            return solution

        def evaluate_constraints(self,solution: FloatSolution) -> None:
            constraints = [0.0 for _ in range(self.number_of_constraints)]

            alpha = solution.variables[0]
            beta = solution.variables[1]
            delta = solution.variables[2]
            gamma = solution.variables[3]

            constraints[0] = (alpha + beta - 1) - tolerance
            constraints[1] = -(alpha + beta - 1) - tolerance
            constraints[2] = (delta + gamma -1) - tolerance
            constraints[3] = -(delta + gamma -1) - tolerance

            overall_constraint_violation = 0.0
            number_of_violated_constraints = 0.0

            for constrain in constraints:
                if constrain > 0.0:
                    overall_constraint_violation += constrain
                    number_of_violated_constraints += 1

            # print("{} {}".format(overall_constraint_violation, number_of_violated_constraints))
            solution.attributes['overall_constraint_violation'] = overall_constraint_violation
            solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

        
        def get_name(self):
            return "Multi Objective CDH"

    return MultiObjectiveCDH()

def get_modified_drs_objective(radius, density, escape_velocity, surface_temperature):
    global interior_score, surface_score

    tolerance = 10e-20
    error = 10e-20


    def f1(x):
        score = interior_score(radius, density,x[0],x[1])
        objective = -score +  modified_inequality_penalty((x[0] + x[1] - 1)) + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error)
        return objective

    def f2(x):
        score = surface_score(escape_velocity, surface_temperature,x[0],x[1])
        objective = -score + modified_inequality_penalty((x[0] + x[1] - 1)) + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error) 
        return objective

    class MultiObjectiveCDH(FloatProblem):
        def __init__(self, rf_path: str = None):
            super(MultiObjectiveCDH, self).__init__(rf_path = rf_path)
            self.number_of_objectives = 2
            self.number_of_variables = 6
            self.number_of_constraints = 0
            self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
            self.obj_labels = ['f(x)', 'f(y)']

            self.lower_bound = [-2.0 for _ in range(self.number_of_variables)]
            self.upper_bound = [2.0 for _ in range(self.number_of_variables)]

            self.lower_bound[-1] = 0.0
            self.upper_bound[-1] = 5.0

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound
        
        def evaluate(self,solution: FloatSolution) -> FloatSolution:
            alpha = solution.variables[0]
            beta = solution.variables[1]
            constant_factor = solution.variables[-1]
            gamma = (constant_factor * alpha * escape_velocity) / radius 
            solution.variables[2] = gamma
            delta = solution.variables[3]

            solution.objectives[0] = f1([alpha, beta])
            solution.objectives[1] = f2([gamma, delta])
            return solution
    return MultiObjectiveCDH()

def get_modified_crs_objective(radius, density, escape_velocity, surface_temperature):
    global interior_score, surface_score

    tolerance = 10e-10
    error = 10e-10


    def f1(x):
        score = interior_score(radius, density,x[0],x[1])
        objective = -score + equality_penalty(x[0] + x[1] + x[-1] -1, tolerance)  + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error)
        return objective

    def f2(x):
        score = surface_score(escape_velocity, surface_temperature,x[0],x[1])
        objective = -score + equality_penalty(x[0] + x[1] + x[-1] -1, tolerance) + inequality_penalty(-x[0],error) + inequality_penalty(x[0]-1,error)+inequality_penalty(-x[1],error)+inequality_penalty(x[1]-1,error) 
        return objective

    class MultiObjectiveCDH(FloatProblem):
        def __init__(self, rf_path: str = None):
            super(MultiObjectiveCDH, self).__init__(rf_path = rf_path)
            self.number_of_objectives = 2
            self.number_of_variables = 6
            self.number_of_constraints = 0
            self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
            self.obj_labels = ['f(x)', 'f(y)']

            self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
            self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

            self.lower_bound[-2] = 0.0
            self.upper_bound[-2] = 5.0

            self.lower_bound[-1] = 0
            self.upper_bound[-1] = 10e-10

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound
        
        def evaluate(self,solution: FloatSolution) -> FloatSolution:
            alpha = solution.variables[0]
            beta = solution.variables[1]
            constant_factor = solution.variables[-2]
            gamma = (constant_factor * alpha * escape_velocity) / radius 
            solution.variables[2] = gamma
            delta = solution.variables[3]
            e = solution.variables[-1]

            solution.objectives[0] = f1([alpha, beta,e])
            solution.objectives[1] = f2([gamma, delta,e])
            return solution

        def evaluate_constraints(self,solution: FloatSolution) -> None:
            constraints = [0.0 for _ in range(self.number_of_constraints)]

            alpha = solution.variables[0]
            beta = solution.variables[1]
            delta = solution.variables[2]
            gamma = solution.variables[3]

            constraints[0] = (alpha + beta -1) - tolerance
            constraints[1] = -(alpha + beta -1) - tolerance
            constraints[2] = (delta + gamma -1) - tolerance
            constraints[3] = -(delta + gamma -1) - tolerance

            overall_constraint_violation = 0.0
            number_of_violated_constraints = 0.0

            for constrain in constraints:
                if constrain > 0.0:
                    overall_constraint_violation += constrain
                    number_of_violated_constraints += 1

            # print("{} {}".format(overall_constraint_violation, number_of_violated_constraints))
            solution.attributes['overall_constraint_violation'] = overall_constraint_violation
            solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints

        
        def get_name(self):
            return "Multi Objective CDH"

    return MultiObjectiveCDH()





def get_ceesa_drs(radius, density, escape_velocity, surface_temperature, orb_eccentricity):
    global ceesa_score

    tolerance = 10e-10
    error = 10e-10


    def f1(x):
        score = ceesa_score(escape_velocity, surface_temperature, radius, density, orb_eccentricity, x[0],x[1], x[2], x[3], x[4], x[5], x[6])
        objective = -score + inequality_penalty(x[0]-1,error) + inequality_penalty(-x[1],error) + inequality_penalty(-x[2],error) + inequality_penalty(-x[3],error) \
            + inequality_penalty(-x[4],error) + inequality_penalty(-x[5],error) + inequality_penalty(-x[6],error) \
            + inequality_penalty(x[1]-1,error) + inequality_penalty(x[2]-1,error) + inequality_penalty(x[3]-1,error) + inequality_penalty(x[4]-1,error) + inequality_penalty(x[5]-1,error) + inequality_penalty(x[6]-1,error) \
            + equality_penalty(x[2] + x[3] + x[4] + x[5] + x[6] -1,tolerance) + equality_penalty(1-(x[6]-x[5]-x[4]-x[3]-x[2]),tolerance)    
        return objective

    class SingleObjectiveCEESA(FloatProblem):
        def __init__(self, rf_path: str = None):
            super(SingleObjectiveCEESA, self).__init__(rf_path = rf_path)
            self.number_of_objectives = 1
            self.number_of_variables = 7
            self.number_of_constraints = 3
            self.obj_directions = [self.MINIMIZE]
            self.obj_labels = ['f(x)']

            self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
            self.upper_bound = [1 - 1e-10 for _ in range(self.number_of_variables)]

            self.lower_bound[0] = 1e-10
            self.upper_bound[0] = 1.0

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound
        
        def evaluate(self,solution: FloatSolution) -> FloatSolution:
            rho = solution.variables[0]
            nu = solution.variables[1]
            e = solution.variables[2]
            t = solution.variables[3] 
            v = solution.variables[4]
            d = solution.variables[5]
            r = solution.variables[6]
            solution.objectives[0] = f1([rho,nu,e,t,v,d,r])
            return solution

        def evaluate_constraints(self,solution: FloatSolution) -> None:
            constraints = [0.0 for _ in range(self.number_of_constraints)]
            rho = solution.variables[0]
            nu = solution.variables[1]
            e = solution.variables[2]
            t = solution.variables[3] 
            v = solution.variables[4]
            d = solution.variables[5]
            r = solution.variables[6]
            constraints[0] = rho - 1
            constraints[1] = r+d+t+v+e -1 -tolerance
            constraints[2] = 1 - (r-d-v-t-e) - tolerance


            overall_constraint_violation = 0.0
            number_of_violated_constraints = 0.0

            for constrain in constraints:
                if constrain > 0.0:
                    overall_constraint_violation += constrain
                    number_of_violated_constraints += 1

            # print("{} {}".format(overall_constraint_violation, number_of_violated_constraints))
            solution.attributes['overall_constraint_violation'] = overall_constraint_violation
            solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints
        def get_name(self):
            return "Single Objective CEESA"

    return SingleObjectiveCEESA()
'''
def get_ceesa_crs(radius, density, escape_velocity, surface_temperature, orb_eccentricity):
    global ceesa_score

    tolerance = 10e-10
    error = 10e-10


    def f1(x):
        score = ceesa_score(escape_velocity, surface_temperature, radius, density, orb_eccentricity, x[0],1, x[1], x[2], x[3], x[4], x[5])
        objective = -score + inequality_penalty(x[0]-1,error) + inequality_penalty(-x[1],error) + inequality_penalty(-x[2],error) + inequality_penalty(-x[3],error) \
            + inequality_penalty(-x[4],error) + inequality_penalty(-x[5],error) + inequality_penalty(-x[6],error) \
            + inequality_penalty(x[1]-1,error) + inequality_penalty(x[2]-1,error) + inequality_penalty(x[3]-1,error) + inequality_penalty(x[4]-1,error) + inequality_penalty(x[5]-1,error) + inequality_penalty(x[6]-1,error) \
            + equality_penalty(x[2] + x[3] + x[4] + x[5] + x[6] -1,tolerance) + equality_penalty(1-(x[6]-x[5]-x[4]-x[3]-x[2]),tolerance)    
        return objective

    class SingleObjectiveCEESA(FloatProblem):
        def __init__(self, rf_path: str = None):
            super(SingleObjectiveCEESA, self).__init__(rf_path = rf_path)
            self.number_of_objectives = 1
            self.number_of_variables = 7
            self.number_of_constraints = 3
            self.obj_directions = [self.MINIMIZE]
            self.obj_labels = ['f(x)']

            self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
            self.upper_bound = [1 - 1e-10 for _ in range(self.number_of_variables)]

            self.lower_bound[0] = 1e-10
            self.upper_bound[0] = 1.0

            FloatSolution.lower_bound = self.lower_bound
            FloatSolution.upper_bound = self.upper_bound
        
        def evaluate(self,solution: FloatSolution) -> FloatSolution:
            rho = solution.variables[0]
            nu = solution.variables[1]
            e = solution.variables[2]
            t = solution.variables[3] 
            v = solution.variables[4]
            d = solution.variables[5]
            r = solution.variables[6]
            solution.objectives[0] = f1([rho,nu,e,t,v,d,r])
            return solution

        def evaluate_constraints(self,solution: FloatSolution) -> None:
            constraints = [0.0 for _ in range(self.number_of_constraints)]
            rho = solution.variables[0]
            nu = solution.variables[1]
            e = solution.variables[2]
            t = solution.variables[3] 
            v = solution.variables[4]
            d = solution.variables[5]
            r = solution.variables[6]
            constraints[0] = rho - 1
            constraints[1] = r+d+t+v+e -1 -tolerance
            constraints[2] = 1 - (r-d-v-t-e) - tolerance


            overall_constraint_violation = 0.0
            number_of_violated_constraints = 0.0

            for constrain in constraints:
                if constrain > 0.0:
                    overall_constraint_violation += constrain
                    number_of_violated_constraints += 1

            # print("{} {}".format(overall_constraint_violation, number_of_violated_constraints))
            solution.attributes['overall_constraint_violation'] = overall_constraint_violation
            solution.attributes['number_of_violated_constraints'] = number_of_violated_constraints
        def get_name(self):
            return "Single Objective CEESA"

    return SingleObjectiveCEESA()
'''