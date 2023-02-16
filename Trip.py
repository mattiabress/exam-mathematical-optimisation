import math
from Point import Point
import copy
class Trip:

    rho_c=6.0
    rho_w=1.39
    omega=100

    def __init__(self, J=[], pi=[], k=0,start_depot=Point(0,0),end_depot=Point(0,0)):
        self.J = J #J' realocation moves
        self.pi = pi #
        self.k = k #number of drop off points
        self.start_depot=start_depot
        self.end_depot=end_depot


    def get_cluster_center(self):
        n = len(self.J)
        if n==0:
            return Point(0.0,0.0)
        sum_u=0
        sum_v=0
        for j in self.J:
            sum_u+=j.u
            sum_v += j.v
        return Point(sum_u/n,sum_v/n)

    def distance_from_cluster_center(self,current_point):
        center=self.get_cluster_center()
        return math.dist([current_point.u,current_point.v], [center.u,center.v])


    def trip_duration(self):
        time_pure_taxi_trip = self.get_pure_taxi_trip()
        trip_time_drop_off_array = self.get_array_pure_taxi_trip_drop_off()
        realocation_move_times = []
        for j in self.J:
            time_plus_realocation = copy.deepcopy(trip_time_drop_off_array)
            for idx, pi_p in enumerate(self.pi[1:-1]):
                realocation_move_time = Trip.get_travel_time_relocation_move(j, pi_p,self.end_depot)
                time_plus_realocation[idx] += realocation_move_time
            if len(time_plus_realocation)>0:
                realocation_move_times.append(min(time_plus_realocation))
        if len(realocation_move_times)==0:
            return max(time_pure_taxi_trip,0)
        return max(time_pure_taxi_trip, max(realocation_move_times))

    def get_pure_taxi_trip(self):
        duration_pure_taxi_trip=0.0
        for i in range(len(self.pi)-1):
            duration_pure_taxi_trip+=Trip.get_travel_time_drop_off(self.pi[i],self.pi[i+1]) #delta_i1_i2
        return duration_pure_taxi_trip

    def get_array_pure_taxi_trip_drop_off(self):
        actual_time=0.0
        duration_pure_taxi_trip_array=[]
        duration_pure_taxi_trip_array.append(actual_time)
        for i in range(len(self.pi)-1): # 0 pi 0,   1
            actual_time+=Trip.get_travel_time_drop_off(self.pi[i],self.pi[i+1]) #delta_i1_i2
            duration_pure_taxi_trip_array.append(actual_time)
        duration_pure_taxi_trip_array=duration_pure_taxi_trip_array[1:-1]
        return duration_pure_taxi_trip_array


    @staticmethod
    def get_travel_time_drop_off(point1,point2):
        travel_time=0.0
        if point1.v!=point2.v:
            return ( min(point1.u+point2.u, 2*Trip.omega-point1.u-point2.u) +abs(point1.v-point2.v) )/Trip.rho_c
        else:
            return abs(point1.u-point2.u)/Trip.rho_c

    @staticmethod
    def get_travel_time_relocation_move(realocation_move, drop_off_point,end_depot):
        return math.sqrt((drop_off_point.u - realocation_move.u) ** 2 + (drop_off_point.v - realocation_move.v) ** 2) /Trip.rho_w +(abs(realocation_move.u - end_depot.u) + abs(realocation_move.v - end_depot.v)) / Trip.rho_c

    def __copy__(self):
        J_new=copy.deepcopy(self.J)
        pi_new=copy.deepcopy(self.pi)
        k_new=copy.deepcopy(self.k)
        start_depot=copy.deepcopy(self.start_depot)
        end_depot = copy.deepcopy(self.end_depot)
        return Trip(J_new,pi_new,k_new,start_depot,end_depot)

    def copy(self):
        return self.__copy__()
    def __str__(self):
        s="J: "
        for j in self.J:
            s+=f'{j} '
        s+="pi: "
        for pi in self.pi:
            s+=f'{pi} '
        s+=f'k={self.k}'
        return s


    @staticmethod
    def getM(m, J, D, start_point, end_point):

        max_distances_start_point = 0.0
        for i in D:
            actual_distance = Trip.get_travel_time_drop_off(start_point, i)
            max_distances_start_point = actual_distance if actual_distance > max_distances_start_point else max_distances_start_point

        max_distances_end_point = 0.0
        for i in D:
            actual_distance = Trip.get_travel_time_drop_off(i, end_point)
            max_distances_end_point = actual_distance if actual_distance > max_distances_end_point else max_distances_end_point

        max_distance_two_points = 0.0
        for i1 in D:
            for i2 in D:
                actual_distance = Trip.get_travel_time_drop_off(i1, i2)
                max_distance_two_points = actual_distance if actual_distance > max_distance_two_points else max_distance_two_points

        max_distance_realocation_move_drop_off = 0.0
        for j in J:
            for i in D:
                actual_distance = Trip.get_travel_time_relocation_move(j, i, end_point)
                max_distance_realocation_move_drop_off = actual_distance if actual_distance > max_distance_realocation_move_drop_off else max_distance_realocation_move_drop_off

        total = max_distances_start_point + max_distances_end_point + (
                    1 - m) * max_distance_two_points + max_distance_realocation_move_drop_off

        return int(total)
