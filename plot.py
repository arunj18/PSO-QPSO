import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os

# with open("qpso_trappist_modified_crs_test_integrated.json", "r") as f:
# 	data = json.loads(f.read())
mypath = "./iteration-wise"
onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
for fi in onlyfiles:

	with open(os.path.join(mypath,fi), "r") as f:
		data = json.loads(f.read())
	
	trappist = filter(lambda x: "TRAPPIST" in x, data.keys())

	for planet in list(data.keys()):
		interior_scores = []
		surface_scores = []
		for front in data[planet]["pareto_front"]:
			interior_scores.append(front["objectives"]["interior_score"])
			surface_scores.append(front["objectives"]["surface_score"])

		print(planet)

		plt.scatter(interior_scores , surface_scores , marker = "x")
		plt.xlabel("interior score")
		plt.ylabel("surface score")
		plt.title("Pareto front for {}".format(planet))
		plt.savefig("./results/{}-".format(planet.replace(" ","_"))+str(fi)+".png")
		plt.close()
