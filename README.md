# Exam of Mathematical Optimisation
## Initialize environment
* `` conda create --name examMO python=3.10``
* `` conda activate examMO ``

* ``conda install --yes ipython jupyterlab``
* ``pip install matplotlib``
* ``pip install numpy``
## Version
* Python 3.10
* Gurobi 9.5.1

## Code
### Classes
* Point: this class represents a Point that is used to define a location of relocation moves or a drop-off point
* Trip: this class represents a Trip and provides some static methods
* Trips: is a static class which gives all methods necessary to work with array of Trip
* Solver: is a static class which provides all methods used in the paper to solve the problem
* Simulation: is a static class it used to prepare the environment

### Simulations
* Project: is a python notebook which is used to test one instance
* scalability_experiments: represents all experiments of scalability

### Extra files:
There are 3 files whose format is .pkl, and they are the result of the three scalability simulations

## References
* https://www.gurobi.com/documentation/9.1/quickstart_mac/cs_using_pip_to_install_gr.html
* https://www.sciencedirect.com/science/article/abs/pii/S0305054820303051#f0045
