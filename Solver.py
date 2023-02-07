import copy
from Trips import Trips
from Trip import Trip
from Point import Point
import numpy as np
import gurobipy as gb

class Solver:

    @staticmethod
    # FIX IT
    # TODO error when n=10, m=1 because n_realocation_moves becomes 0 so all trips have 0 reallocation moves except the last one which has all reallocation moves
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
            # print(t)
            realocation_moves = copy.deepcopy(J)
            drop_off_points = copy.deepcopy(D)

            # set initial current state sc
            n_trips = int(np.ceil(n / (np.ceil((m + 1) / 2))))
            trips = []

            for i in range(n_trips):
                k = np.random.randint(1, m + 1)  # number of drop-off points
                pi = np.random.choice(drop_off_points, k, replace=False)  # These drop-off points are randomly chosen
                pi = pi.tolist()

                # end and start depot
                pi.insert(0, start_point)
                pi.append(end_point)

                # the relocation moves are randomly spread between the different trips.

                n_realocation_moves = int(len(J) / n_trips)  # TODO we can improve using the normal distribution to select the right dimension

                J_prime = np.random.choice(realocation_moves, n_realocation_moves, replace=False)
                J_prime = J_prime.tolist()
                if i == n_trips - 1:
                    J_prime = copy.deepcopy(realocation_moves)
                for j in J_prime:
                    realocation_moves.remove(j)

                trip = Trip(J_prime, pi, k)
                trips.append(trip)
                # Finally, for each trip and its assigned relocation moves, we execute each relocation from the closest drop-off point increasing the trip duration least.

                # if I understood well, it means we have to take the closest drop off point to the relocation moves to select in the time duration

            # initialize the optimal solution
            if t==0:
                trips_star = copy.deepcopy(trips)

            for j in range(kr):
                T = T_start
                not_improved = 0
                new_trips = copy.deepcopy(trips)
                while not_improved < kn and T != 0:
                    not_improved += 1

                    # Sn= neighbour of current state Sc
                    selected_indexes_trips = np.random.choice(range(len(new_trips)), 2, replace=False)
                    selected_indexes_trips = selected_indexes_trips.tolist()
                    trip1 = copy.deepcopy(new_trips[selected_indexes_trips[0]])
                    trip2 = copy.deepcopy(new_trips[selected_indexes_trips[1]])

                    # select an action
                    n_actions = np.random.randint(0, 5)
                    match n_actions:
                        case 0:
                            # Swap two relocation moves between two taxi trips
                            Trips.swap_random_realocation_moves(trip1, trip2)
                        case 1:
                            # Move a relocation move to another taxi trip
                            Trips.move_random_realocation_moves(trip1, trip2)

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
                                1.0 / exp_lambda)  # TODO sistemare calcolo probabilità
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
            new_trip = Trip([], [], 0)

            p = np.floor(n / m) / (np.floor(n / m) + 1)

            if np.random.uniform(0, 1) <= p:
                n_hat = np.random.randint(np.ceil((m + 1) / 2), m + 1)
            else:
                n_hat = np.random.randint(1, np.ceil((m + 1) / 2) + 1)

            j = realocation_moves[np.random.randint(0, len(realocation_moves))]  # len(J) = |J|
            # update
            new_trip.J.append(j)  # J'={j}
            realocation_moves.remove(j)  # J=J\{j}   P= (u_j^c,v_j^c)

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
            new_trip.pi.append(pi_l)  # pi={pi_l} #TODO  sistemare con un metodo
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
    def sam_matheuristic(n, m,J, D, trips):
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
                    realocation_time = Trip.get_travel_time_relocation_move(J[j], pi_p)
                    time_plus_realocation[idx] += realocation_time
                delta[i, j] = max(0, min(time_plus_realocation) - C[i])

        sam_mip = gb.Model()
        sam_mip.modelSense = gb.GRB.MINIMIZE  # declare mimization
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
    def local_search(trips, J, D, kn):
        trips_current_solution = copy.deepcopy(trips)
        not_improved = 0
        while not_improved < kn:
            not_improved += 1
            new_trips = copy.deepcopy(trips_current_solution)

            # select current neighborhood with equal probability
            selected_indexes_trips = np.random.choice(range(len(new_trips)), 2, replace=False)
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
                    Trips.move_random_realocation_moves(trip1, trip2)

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
