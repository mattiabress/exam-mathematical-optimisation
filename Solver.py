import copy
from Trips import Trips
from Trip import Trip
from Point import Point
import numpy as np
import gurobipy as gb


class Solver:

    @staticmethod
    def sa_approach(n, m, ks, kr, kn, T_start, c, J, D, start_point, end_point):
        # ks -> number of restarts
        # kr -> number of rehats
        # T_start -> initial temperature
        # c -> cooling parameter
        # J -> realocation moves
        # D -> drop off points
        # By varying these parameters and caching not only the best solution found so far, but also the temporary best solution, i.e., before reheat and restart
        # we obtain a pool of taxi trips by splitting the cached solutions, i.e., taxi tour, into the single trips

        for t in range(ks):
            # generate random feasible solution
            realocation_moves = copy.deepcopy(J)
            drop_off_points = copy.deepcopy(D)

            # set initial current state sc
            trips = Solver.cc_procedure(n, m, J, D, start_point, end_point)


            # initialize the optimal solution
            if t == 0:
                trips_star = copy.deepcopy(trips)

            for j in range(kr):
                T = T_start
                not_improved = 0
                new_trips = copy.deepcopy(trips)
                while not_improved < kn and T != 0:
                    not_improved += 1

                    # Sn= neighbour of current state Sc
                    if len(new_trips) > 1:
                        selected_indexes_trips = np.random.choice(range(len(new_trips)), 2, replace=False)
                        selected_indexes_trips = selected_indexes_trips.tolist()
                        trip1 = copy.deepcopy(new_trips[selected_indexes_trips[0]])
                        trip2 = copy.deepcopy(new_trips[selected_indexes_trips[1]])
                    else:
                        selected_indexes_trips = [0, 0]
                        trip1 = new_trips[selected_indexes_trips[0]].copy()
                        trip2 = new_trips[selected_indexes_trips[1]].copy()

                    # select current neighborhood with equal probability
                    # select an action
                    n_actions = np.random.randint(0, 5)
                    match n_actions:
                        case 0:
                            # Swap two relocation moves between two taxi trips
                            Trips.swap_random_realocation_moves(trip1, trip2)
                        case 1:
                            # Move a relocation move to another taxi trip
                            Trips.move_random_realocation_moves(trip1, trip2,m)

                        case 2:
                            # Remove a drop-off point from a taxi trip
                            Trips.remove_random_dropoff(trip1)

                        case 3:
                            # Swap two drop-off points between two taxi trips
                            Trips.swap_random_dropoff(trip1, trip2)

                        case 4:
                            # Add a drop-off point to a taxi trip
                            pi = np.random.choice(D, 1)[0]
                            Trips.add_drop_off(trip1, pi)

                    new_trips[selected_indexes_trips[0]] = trip1
                    new_trips[selected_indexes_trips[1]] = trip2

                    z_trips = Trips.get_total_duration(trips)
                    z_new_trips = Trips.get_total_duration(new_trips)

                    if (z_new_trips < z_trips):
                        # sc=sn update current
                        trips = new_trips
                        z_trips = z_new_trips
                        not_improved = 0
                        z_trips_star = Trips.get_total_duration(trips_star)
                        if (z_trips < z_trips_star):
                            trips_star = copy.deepcopy(trips)
                    else:
                        # with probability exp() assign sc=sn
                        exp_lambda = (z_new_trips - z_trips) / T
                        if abs(exp_lambda) > np.finfo(float).eps:
                            exponential_sample = np.random.exponential(
                                1.0 / exp_lambda)
                            trips = new_trips if exponential_sample < 1.0 / exp_lambda else trips  # sc=sn if probability else sc
                    T = T * c
        return trips_star

    @staticmethod
    #  customized clustering procedure
    def cc_procedure(n, m, J, D, start_point, end_point):
        # n is number of realocation moves J={1, ..., n}
        # m is number of realocators R={1, ..., m}

        realocation_moves = copy.deepcopy(J)
        trips = []

        while len(realocation_moves) > 0:
            drop_off_points = copy.deepcopy(D)
            # initialize a new Trip
            new_trip = Trip([], [], 0,start_point,end_point)

            p = np.floor(n / m) / (np.floor(n / m) + 1)

            if np.random.uniform(0, 1) <= p:
                n_hat = np.random.randint(np.ceil((m + 1) / 2), m + 1)
            else:
                n_hat = np.random.randint(1, np.ceil((m + 1) / 2) + 1)

            j = realocation_moves[np.random.randint(0, len(realocation_moves))]  # len(J) = |J|
            # update
            new_trip.J.append(j)  # J'={j}
            realocation_moves.remove(j)  # J=J\{j}

            while len(new_trip.J) < n_hat and len(realocation_moves) != 0:
                p = []
                for j in realocation_moves:
                    p_j = 0.0
                    for q in J:
                        distance_j = new_trip.distance_from_cluster_center(j)
                        distance_q = new_trip.distance_from_cluster_center(q)
                        p_j += distance_j / distance_q if distance_q != 0 else 0
                    p.append(p_j ** -1 if p_j != 0 else 0.0)
                j = np.random.choice(realocation_moves, 1, p)[0]  # sample from J
                new_trip.J.append(j)  # J'={j}
                realocation_moves.remove(j)  # J=J\{j}

            # line 7
            probability = []
            for selected_pi in drop_off_points:
                p_pi = 0.0
                for q in drop_off_points:
                    distance_pi = new_trip.distance_from_cluster_center(selected_pi)
                    sum_distance_dropoff = new_trip.distance_from_cluster_center(q)
                    p_pi += distance_pi / sum_distance_dropoff if sum_distance_dropoff != 0 else 0
                probability.append(p_pi ** -1 if p_pi != 0 else 0.0)

            pi_l = np.random.choice(drop_off_points, 1, probability)[0]

            new_trip.pi.append(start_point)
            new_trip.pi.append(pi_l)  # pi={pi_l}
            new_trip.pi.append(end_point)
            new_trip.k = 1

            drop_off_points.remove(pi_l)  # D=D\{pi_l}
            # calculate total duration C of taxi trip pi according to relocation J'
            C = new_trip.trip_duration()
            while len(drop_off_points) != 0:
                # determine drop-off point q in D which decreases C most
                q_min = None
                for q in drop_off_points:
                    new_trip.pi.insert(len(new_trip.pi) - 1, q)
                    C_new = new_trip.trip_duration()
                    if C_new < C:
                        q_min = q
                    new_trip.pi.remove(q)

                if (q_min != None):
                    new_trip.pi.insert(len(new_trip.pi) - 1, q_min)
                    new_trip.k += 1
                    drop_off_points.remove(q_min)
                else:
                    drop_off_points = []

            trips.append(new_trip)

        return trips

    @staticmethod
    def sam_matheuristic(n, m, J, D, trips,time_limit=60):
        # I-> trips
        I = range(len(trips))

        # preparate the variables
        C = []
        for i in I:
            C.append(trips[i].get_pure_taxi_trip())

        delta = np.zeros((len(trips), len(J)))

        for i in I:  # trips
            trip_time_drop_off_array = trips[i].get_array_pure_taxi_trip_drop_off()
            for j in range(len(J)):
                time_plus_realocation = copy.deepcopy(trip_time_drop_off_array)
                for idx, pi_p in enumerate(trips[i].pi):
                    realocation_time = Trip.get_travel_time_relocation_move(J[j], pi_p,trips[i].end_depot)
                    time_plus_realocation[idx] += realocation_time
                delta[i, j] = max(0, min(time_plus_realocation) - C[i])

        # initialize the Gurobi problem
        sam_mip = gb.Model()
        sam_mip.modelSense = gb.GRB.MINIMIZE  # declare mimization
        sam_mip.setParam(gb.GRB.Param.TimeLimit, time_limit)
        X = sam_mip.addVars([(i, j) for i in I for j in range(len(J))], vtype=gb.GRB.BINARY)
        Y = sam_mip.addVars([i for i in I], vtype=gb.GRB.BINARY)
        b = sam_mip.addVars([i for i in I], lb=0, vtype=gb.GRB.CONTINUOUS)

        # Xi,j==1
        for j in range(len(J)):
            sam_mip.addConstr(gb.quicksum(X[i, j] for i in I) == 1)
        # X(i,j)<=Y(i)*m
        for i in I:
            sam_mip.addConstr(gb.quicksum(X[i, j] for j in range(len(J))) <= Y[i] * m)

        for i in I:
            for j in range(len(J)):
                # b[i]>=X[i,j]*delta[i,j]
                sam_mip.addConstr(b[i] >= X[i, j] * delta[i, j])  # delta(i,j)

        sam_mip.setObjective(gb.quicksum(Y[i] * C[i] + b[i] for i in I))
        sam_mip.optimize()

        # print( "\n", type(X), X, "\n")
        print("\nSolution")

        print("Binary variables: 1, if relocation move j in J is executed on taxi trip i in I;0, otherwise")
        for i in I:
            for j in range(len(J)):
                if X[i, j].x == 1:  # to access the variable value
                    print(f'realocation move {j} is executed on taxi trip {i}')

        print("Binary variables: 1, if taxi trip i in I is selected from the pool; 0,otherwise")
        for i in I:
            if Y[i].x == 1:  # to access the variable value
                print(f'taxi trip {i} is selected from the pool')

        copied_trips = copy.deepcopy(trips)
        realocation_moves = copy.deepcopy(J)
        for i in I:
            copied_trips[i].J = []
            for j in range(len(J)):
                if X[i, j].x == 1:
                    copied_trips[i].J.append(realocation_moves[j])

        new_trips = [copied_trips[idx] for idx in range(len(copied_trips)) if Y[idx].x == 1]

        return new_trips

    @staticmethod
    def sm_matheuristic(J, D, trips,time_limit=60):
        I = range(len(trips))
        # to construct theta
        theta = np.zeros((len(trips), len(J)))
        for i in I:
            for j in range(len(J)):
                theta[i, j] = 1 if any(obj.u == J[j].u and J[j].v for obj in trips[i].J) else 0

        C = []
        for trip in trips:
            time_pure_taxi_trip = trip.get_pure_taxi_trip()
            trip_time_drop_off_array = trip.get_array_pure_taxi_trip_drop_off()
            realocation_move_time_array = []

            for j in trip.J:
                time_plus_realocation = copy.deepcopy(trip_time_drop_off_array)
                for idx, pi_p in enumerate(trip.pi):
                    realocation_move_time = Trip.get_travel_time_relocation_move(j, pi_p,trip.end_depot)
                    time_plus_realocation[idx] += realocation_move_time

                realocation_move_time_array.append(min(time_plus_realocation))
            if realocation_move_time_array != []:
                time_result = max(time_pure_taxi_trip, max(realocation_move_time_array))
            else:
                time_result = max(time_pure_taxi_trip, 0)
            C.append(time_result)

        sm_mip = gb.Model()
        sm_mip.modelSense = gb.GRB.MINIMIZE  # declare mimization
        sm_mip.setParam(gb.GRB.Param.TimeLimit, time_limit)
        Y = sm_mip.addVars([i for i in I], vtype=gb.GRB.BINARY)
        for j in range(len(J)):
            sm_mip.addConstr(gb.quicksum(Y[i] * theta[i, j] for i in I) >= 1)

        sm_mip.setObjective(gb.quicksum(Y[i] * C[i] for i in I))
        sm_mip.optimize()

        # print( "\n", type(Y), Y, "\n")
        # print("\nSolution")
        #
        # for i in I:
        #     if Y[i].x==1:
        #         print(f'Y[{i}]= {Y[i].x}')

        copied_trips = copy.deepcopy(trips)
        new_trips = [copied_trips[idx] for idx in range(len(copied_trips)) if Y[idx].x == 1]

        return new_trips

    @staticmethod
    def local_search(n,m,J, D, kn,trips):
        trips_current_solution = copy.deepcopy(trips)
        not_improved = 0
        while not_improved < kn:
            not_improved += 1
            new_trips = copy.deepcopy(trips_current_solution)

            # select current neighborhood with equal probability
            if len(new_trips) > 1:
                selected_indexes_trips = np.random.choice(range(len(new_trips)), 2, replace=False)
                trip1 = new_trips[selected_indexes_trips[0]].copy()
                trip2 = new_trips[selected_indexes_trips[1]].copy()
            else:
                selected_indexes_trips=[0,0]
                trip1 = new_trips[selected_indexes_trips[0]].copy()
                trip2 = new_trips[selected_indexes_trips[1]].copy()

            # select an action
            n_actions = np.random.randint(0, 5)
            match n_actions:
                case 0:
                    # Swap two relocation moves between two taxi trips
                    Trips.swap_random_realocation_moves(trip1, trip2)
                case 1:
                    # Move a relocation move to another taxi trip
                    Trips.move_random_realocation_moves(trip1, trip2,m)

                case 2:
                    # Remove a drop-off point from a taxi trip
                    Trips.remove_random_dropoff(trip1)

                case 3:
                    # Swap two drop-off points between two taxi trips
                    Trips.swap_random_dropoff(trip1, trip2)

                case 4:
                    # Add a drop-off point to a taxi trip
                    pi = np.random.choice(D, 1)[0]
                    Trips.add_drop_off(trip1, pi)

            new_trips[selected_indexes_trips[0]] = trip1
            new_trips[selected_indexes_trips[1]] = trip2

            z_trips = Trips.get_total_duration(trips_current_solution)
            z_new_trips = Trips.get_total_duration(new_trips)

            if (z_new_trips < z_trips):
                # sc=sn update current
                trips_current_solution = copy.deepcopy(new_trips)
                not_improved = 0
        return trips_current_solution


    @staticmethod
    def trptr_problem(n, m, realocation_moves, drop_off_points, start_point, end_point,time_limit=300):
        K = range(0, int(np.ceil(n / (np.ceil((m + 1) / 2)))))
        R = range(0, m)
        J = range(0, n)
        D = range(len(drop_off_points))
        central_depot_s = start_point
        central_depot_e = end_point
        M = Trip.getM(m,realocation_moves,drop_off_points,start_point,end_point)

        trptr_mip = gb.Model()
        trptr_mip.modelSense = gb.GRB.MINIMIZE
        trptr_mip.setParam(gb.GRB.Param.TimeLimit, time_limit)

        # Variables
        t = trptr_mip.addVars([(k, i) for k in K for i in D], lb=0, vtype=gb.GRB.CONTINUOUS)
        C = trptr_mip.addVars([k for k in K], lb=0, vtype=gb.GRB.CONTINUOUS)
        S = trptr_mip.addVars([(k, i, p) for k in K for i in D for p in R], vtype=gb.GRB.BINARY)
        Y = trptr_mip.addVars([(j, i) for j in J for i in D], vtype=gb.GRB.BINARY)
        X = trptr_mip.addVars([(k, j) for k in K for j in J], vtype=gb.GRB.BINARY)

        # Contraints

        # X(k,j)=1
        for j in J:
            trptr_mip.addConstr(gb.quicksum(X[k, j] for k in K) == 1)
        # Y(j,i)=1
        for j in J:
            trptr_mip.addConstr(gb.quicksum(Y[j, i] for i in D) == 1)
        # X[k,j]<=m
        for k in K:
            trptr_mip.addConstr(gb.quicksum(X[k, j] for j in J) <= m)

        # X[k,j]+Y[j,i]<=1+sum_p S[k,i,p]
        for k in K:
            for j in J:
                for i in D:
                    trptr_mip.addConstr(X[k, j] + Y[j, i] <= 1 + gb.quicksum(S[k, i, p] for p in R))



        # Contraint on S[k,i,p]

        # S[k,i,p]<=1
        for k in K:
            for p in R:
                trptr_mip.addConstr(gb.quicksum(S[k, i, p] for i in D) <= 1)

        # S[k,i,p]<=1
        for k in K:
            for i in D:
                trptr_mip.addConstr(gb.quicksum(S[k, i, p] for p in R) <= 1)

        # S[k,i,p]<=S[k,i,p-1]
        for k in K:
            for p in [x for x in R if x != 0]:
                trptr_mip.addConstr(gb.quicksum(S[k, i, p] for i in D)  <= gb.quicksum(S[k, i, p - 1] for i in D))

        # t's constraints
        # t[k,i]>=S[k,i,0]*delta[start,i]
        for k in K:
            for i in D:
                trptr_mip.addConstr(
                    t[k, i] >= S[k, i, 0] * Trip.get_travel_time_drop_off(central_depot_s, drop_off_points[i]))

        for k in K:
            for i1 in D:
                for i2 in [x for x in D if x != i1]:  # D\{i1}
                    for p in [x for x in R if x != 0]:
                        # trptr_mip.addConstr(t[k,i2]>=t[k,i1]-M*(2-S[k,i1,p-1]-S[k,i2,p])+delta[i1,i2])
                        trptr_mip.addConstr(t[k, i2] >= t[k, i1] - M * (
                                    2 - S[k, i1, p - 1] - S[k, i2, p]) + Trip.get_travel_time_drop_off(
                            drop_off_points[i1], drop_off_points[i2]))

        # C's constraints
        for k in K:
            for j in J:
                for i in D:
                    # trptr_mip.addConstr(C[k]>=t[k,i]-M*(2-X[k,j]-Y[j,i])+d[j,i])
                    trptr_mip.addConstr(
                        C[k] >= t[k, i] - M * (2 - X[k, j] - Y[j, i]) + Trip.get_travel_time_relocation_move(
                            realocation_moves[j], drop_off_points[i],end_point))

        for k in K:
            for i in D:
                trptr_mip.addConstr(C[k] >= t[k, i] - M * (1 - gb.quicksum(S[k, i, p] for p in R)) + Trip.get_travel_time_drop_off(
                        drop_off_points[i], central_depot_e))
        # C[k]<C[k-1]
        for k in [x for x in K if x != 0]:
            trptr_mip.addConstr(C[k] <= C[k - 1])

        # Objective function
        trptr_mip.setObjective(gb.quicksum(C[k] for k in K))
        # Solution
        trptr_mip.optimize()
        # print("\nSolution")
        #
        # for k in K:
        #     for i in D:
        #         if t[k,i].x!=0:
        #             print(f't[{k},{i}]={t[k,i].x}  drop_off={drop_off_points[i]}')
        # print("")
        # for k in K:
        #     for p in R:
        #         for i in D:
        #             if S[k,i,p].x==1:
        #                 print(f' S[{k},{i},{p}]=1 dropoff={drop_off_points[i]}')
        # print("")
        # for j in J:
        #     for i in D:
        #         if Y[j,i].x==1:
        #             print(f'Y[{j},{i}]=1  realocation move={realocation_moves[j]}, dropoff={drop_off_points[i]}')
        # print("")
        # for k in K:
        #     print(f'C[{k}]={C[k].x}')

        trips = []

        for k in K:
            trip = Trip([], [], 0,start_point,end_point)
            for j in J:
                if X[k, j].x == 1:
                    trip.J.append(realocation_moves[j])
            for p in R:
                for i in D:
                    if S[k, i, p].x == 1:
                        trip.pi.append(drop_off_points[i])
                        trip.k += 1
            trip.pi.insert(0, start_point)
            trip.pi.append(end_point)
            trips.append(trip)

        return trips
