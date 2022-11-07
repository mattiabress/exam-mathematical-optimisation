import gurobipy as gp
from gurobipy import GRB
import numpy as np
import scipy.sparse as sp


class Solver:
    @staticmethod
    def solveMILP(parameters, tasks_data):
        # tasks
        J = tasks_data.shape[1]
        u = tasks_data[0]
        q = tasks_data[1]
        y_min = tasks_data[2]
        y_max = tasks_data[3]
        t_min = tasks_data[4]
        t_max = tasks_data[5]
        p_min = tasks_data[6]
        p_max = tasks_data[7]
        w_min = tasks_data[8]
        w_max = tasks_data[9]

        # parameters
        if J != parameters[0]:
            raise Exception("The number of tasks is wrong")
        T = parameters[1]
        Q = parameters[2]
        Vb = parameters[3]
        e = parameters[4]
        SoC_0 = parameters[5]
        rho = parameters[6]
        gamma = parameters[7]

        try:
            # Create a new model
            m = gp.Model("MILP")

            # Create variables
            x = m.addMVar(shape=3, vtype=GRB.BINARY, name="x")

            # Set objective
            obj = np.array([1.0, 1.0, 2.0])
            m.setObjective(obj @ x, GRB.MAXIMIZE)

            # Build (sparse) constraint matrix
            val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
            row = np.array([0, 0, 0, 1, 1])
            col = np.array([0, 1, 2, 0, 1])

            A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

            # Build rhs vector
            rhs = np.array([4.0, -1.0])

            # Add constraints
            m.addConstr(A @ x <= rhs, name="c")

            # Optimize model
            m.optimize()

            print(x.X)
            print('Obj: %g' % m.objVal)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')

    @staticmethod
    def solveUsingCG():
        print("solved MILP using CG algorithm")

    @staticmethod
    def solveRMP(parameters, tasks_data):
        # tasks
        J = tasks_data.shape[1]
        u = tasks_data[0]
        q = tasks_data[1]
        y_min = tasks_data[2]
        y_max = tasks_data[3]
        t_min = tasks_data[4]
        t_max = tasks_data[5]
        p_min = tasks_data[6]
        p_max = tasks_data[7]
        w_min = tasks_data[8]
        w_max = tasks_data[9]

        # parameters
        if J != parameters[0]:
            raise Exception("The number of tasks is wrong")
        T = parameters[1]
        Q = parameters[2]
        Vb = parameters[3]
        e = parameters[4]
        SoC_0 = parameters[5]
        rho = parameters[6]
        gamma = parameters[7]

        try:
            # Create a new model
            m = gp.Model("RMP")

            # Create variables
            x = m.addMVar(shape=J, lb=0, vtype=GRB.CONTINUOUS, name="epsilon")

            # Set objective
            obj = np.array([1.0, 1.0, 2.0])
            m.setObjective(obj @ x, GRB.MAXIMIZE)

            # Build (sparse) constraint matrix
            val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
            row = np.array([0, 0, 0, 1, 1])
            col = np.array([0, 1, 2, 0, 1])

            A = sp.csr_matrix((val, (row, col)), shape=(2, 3))

            # Build rhs vector
            rhs = np.array([4.0, -1.0])

            # Add constraints
            m.addConstr(A @ x <= rhs, name="c")

            # Optimize model
            m.optimize()

            print(x.X)
            print('Obj: %g' % m.objVal)

        except gp.GurobiError as e:
            print('Error code ' + str(e.errno) + ": " + str(e))

        except AttributeError:
            print('Encountered an attribute error')

    @staticmethod
    def solvePS():
        print("solved MILP using CG algorithm")

    @staticmethod
    def getSchedulesForTask(thaskId):
        print(thaskId)
