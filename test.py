from jmetal.algorithm import SMPSO, MOQPSO
from jmetal.component import CrowdingDistanceArchive
from jmetal.problem import DTLZ1
from jmetal.operator import Polynomial,SBX, BinaryTournamentSelection, Uniform
from objective_functions import get_crs_objective, get_drs_objective, get_modified_drs_objective, get_modified_crs_objective,interior_score, surface_score,get_ceesa_drs,ceesa_score
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np
import json

# trappist-1 h
# radius = 0.92
# density =  0.82
# escape_velocity = 0.83
# surface_temperature = 260.4/288.0

# radius = 1.83
# density = 1.19
# escape_velocity = 1.99
# surface_temperature = 0.95

# problem = get_crs_objective(radius, density, escape_velocity, surface_temperature)

with open("trappist.txt") as f:
    planets = f.read().split("\n")

df = pd.read_csv("phl.csv")
df = df[df["P. Name"].isin(planets)]
df = df[df["P. Ts Mean (K)"].isnull() == False]
p_names = df["P. Name"].tolist()
radii = df["P. Radius (EU)"].tolist()
densities = df["P. Density (EU)"].tolist()
esc_velocities = df["P. Esc Vel (EU)"].tolist()
surf_temps = (df["P. Ts Mean (K)"]/ 288.0).tolist()
orb_eccentricities = df["P. Eccentricity"].tolist()
results = {}
complete_results = {}
print("Number of planets {}".format(df.shape[0]))
for eval in [200 for x in range(1)]:
    #print(eval)
    empty_list = []
    for i in range(df.shape[0]):

        p_name = p_names[i]

        radius = radii[i]
        density = densities[i]
        escape_velocity = esc_velocities[i]
        surface_temperature = round(surf_temps[i],2)
        orb_eccentricity = orb_eccentricities[i]

        # p_name = "Earth"
        # radius = 1.0
        # density = 1.0
        # escape_velocity = 1.0
        # surface_temperature = 1.0


        if(p_name not in list(complete_results.keys())):
            complete_results[p_name] = []
        print("")
        print("optimizing for planet : {}".format(p_name))
        print("radius : {}".format(radius))
        print("density: {}".format(density))
        print("escape velocity: {}".format(escape_velocity))
        print("surface temperature : {}".format(surface_temperature))
        print("eccentricity : {}".format(orb_eccentricity))
        print("")

        problem = get_crs_objective(radius, density, escape_velocity, surface_temperature)
        
        # problem = get_modified_drs_objective(radius, density, escape_velocity,surface_temperature)

        algorithm = MOQPSO(
            problem=problem,
            swarm_size=1000,
            max_evaluations=eval,
            mutation=Polynomial(probability=0.2, distribution_index=10),
            leaders=CrowdingDistanceArchive(100),
            reference_point=  [0] * problem.number_of_objectives ,
            levy = 1,
            levy_decay= 1
        )
        # #algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

        # algorithm = SMPSO(
        #         problem=problem,
        #         swarm_size=25,
        #         max_evaluations=100,
        #         mutation=Polynomial(probability=0.2, distribution_index=10),
        #         leaders=CrowdingDistanceArchive(100),
        #         reference_point=[0] * problem.number_of_objectives
        #     )
    
        

        # try:
        algorithm.run()
        front = algorithm.get_result()
        complete_results[p_name].append(algorithm.evaluations)
        print("evaluations",algorithm.evaluations)
        print("hypervolume",algorithm.get_hypervolume_history())
        #input()
        pareto_results = []
        for j in range(len(front)):
            # if(abs(front[i].variables[0] + front[i].variables[1] - 1) > 0.1 or abs(front[i].variables[2] + front[i].variables[3] - 1) > 0.1 ):
            #     continue
            
            pareto_results.append(
                    {
                        "objectives" : { "interior_score" : interior_score(radius,density, front[j].variables[0], front[j].variables[1]),
                                        "surface_score" : surface_score(escape_velocity, surface_temperature,  front[j].variables[2], front[j].variables[3]),
                                        },
                        "variables" : {"alpha" : front[j].variables[0], 
                                        "beta" : front[j].variables[1],
                                        "delta" : front[j].variables[2],
                                        "gamma" : front[j].variables[3],
                                        "C" : front[j].variables[4],
                                        #"e" : front[j].variables[5]
                                    }
                    }

                )
            #print(pareto_results)
        
        if len(pareto_results) == 0:
            print("{} is empty".format(p_names[i]))
            empty_list.append(p_names[i])
        results[p_name] = { 
                            "pareto_front": pareto_results 
                        }
    # hyper_volume = HyperVolume(reference_point = [1.0,1.0])
            # hv = hyper_volume.compute(front)
            # print("Hyper volume: {}".format(hv))
        # except:
        #    print("ERROR")
        #    continue
            
        # print("Number of iterations: {}".format(algorithm.evaluations))
    # print(json.dumps(results, indent = 4))

    with open("./iteration-wise/qpso_trappist-cdhs-crs-iteration"+str(eval)+".json", "w") as f:
        f.write(json.dumps(results, indent=4))


# pareto_front = FrontPlot(plot_title='MOQPSO-DTLZ1-5', axis_labels=problem.obj_labels)
# pareto_front.plot(front, reference_front=problem.reference_front)
# pareto_front.to_html(filename='MOQPSO-DTLZ1-5') 
for i in list(complete_results.keys()):
    complete_results[i] = sum(complete_results[i])/30
print(complete_results)