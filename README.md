<p align="center">
  <br/>
  <img src=docs/source/jmetalpy.png alt="jMetalPy">
  <br/>
</p>

# jMetalPy: Python version of the jMetal framework
[![Build Status](https://img.shields.io/travis/jMetal/jMetalPy.svg?style=flat-square)](https://travis-ci.org/jMetal/jMetalPy)
[![Read the Docs](https://img.shields.io/readthedocs/jmetalpy.svg?style=flat-square)](https://readthedocs.org/projects/jmetalpy/)
[![PyPI License](https://img.shields.io/pypi/l/jMetalPy.svg?style=flat-square)]()
[![PyPI Python version](https://img.shields.io/pypi/pyversions/jMetalPy.svg?style=flat-square)]()

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [License](#license)

## Installation
To download jMetalPy just clone the Git repository hosted in GitHub:
```bash
$ git clone https://github.com/jMetal/jMetalPy.git
$ python setup.py install
```

Alternatively, you can install it with `pip`:
```bash
$ pip install jmetalpy
```

## Usage
Examples of configuring and running all the included algorithms are located [in the docs](https://jmetalpy.readthedocs.io/en/latest/examples.html).

## Features
The current release of jMetalPy (v0.5.1) contains the following components:

* Algorithms: random search, [NSGA-II](https://jmetalpy.readthedocs.io/en/latest/examples/ea.html#nsga-ii-with-plotting), [SMPSO](https://jmetalpy.readthedocs.io/en/latest/examples/pso.html#smpso-with-standard-settings), [SMPSO/RP](https://jmetalpy.readthedocs.io/en/latest/examples/pso.html#smpso-rp-with-standard-settings).
* Benchmark problems: ZDT1-6, DTLZ1-2, unconstrained (Kursawe, Fonseca, Schaffer, Viennet2), constrained (Srinivas, Tanaka).
* Encodings: real, binary.
* Operators: selection (binary tournament, ranking and crowding distance, random, nary random, best solution), crossover (single-point, SBX), mutation (bit-blip, polynomial, uniform, random).
* Quality indicators: [hypervolume](https://jmetalpy.readthedocs.io/en/latest/api/jmetal.component.html#module-jmetal.component.quality_indicator).
* Density estimator: crowding distance.
* Laboratory: [Experiment class for performing studies](https://jmetalpy.readthedocs.io/en/latest/examples/experiment.html).
* Graphics: Pareto front plotting for problems with two or more objectives (as scatter plot/parallel coordinates).

<p align="center">
  <br/>
  <img src=docs/source/2D.gif width=600 alt="Scatter plot 2D">
  <br/>
  <img src=docs/source/3D.gif width=600 alt="Scatter plot 3D">
  <br/>
  <img src=docs/source/p-c.gif width=600 alt="Parallel coordinates">
  <br/>
</p>

## License
This project is licensed under the terms of the MIT - see the [LICENSE](LICENSE) file for details.

** EDITS

Chaotic Quantum PSO is one of the algorithms along with a lorenz map initialization. They've been added to the algorithm part of MOQPSO. The chaos initialization has been added to core under problem.py.

Example use as in test.py: 

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