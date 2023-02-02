import math
import Point
import copy
class Trip:

    rho_c=6.0
    rho_w=1.39
    omega=100

    def __init__(self, J=[], pi=[], k=0):
        self.J = J #J' realocation moves
        self.pi = pi #
        self.k = k #number of drop off points


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

    def duration(self):
        #TODO fare
        return 10

    def trip_duration(self):
        sum_1=0.0
        for i in range(len(self.pi)-1): # TODO Controllare sia meno 1
            sum_1+=Trip.get_travel_time_drop_off(self.pi[i],self.pi[i+1]) #delta_i1_i2
        sum_2=0.0 #TODO sistemare
        return max(sum_1,sum_2)

    def get_pure_taxi_trip(self):
        duration_pure_taxi_trip=0.0
        for i in range(len(self.pi)-1):
            duration_pure_taxi_trip+=Trip.get_travel_time_drop_off(self.pi[i],self.pi[i+1]) #delta_i1_i2
        return duration_pure_taxi_trip


    @staticmethod
    def get_travel_time_drop_off(point1,point2):
        travel_time=0.0
        if point1.v!=point2.v:
            return (min(point1.u+point2.u,2*Trip.omega-point1.u-point2.u)+abs(point1.v-point2.v))/Trip.rho_c
        else:
            return abs(point1.u-point2.u)/Trip.rho_c

    @staticmethod
    def get_travel_time_relocation_move(point1,point2):
        return math.sqrt((point1.u-point2.u)**2+(point1.v-point2.v)**2)/Trip.rho_w+(abs(point1.u-point2.u)+abs(point1.v-point2.v))/Trip.rho_c

    def __copy__(self):
        J_new=copy.deepcopy(self.J)
        pi_new=copy.deepcopy(self.pi)
        k_new=copy.deepcopy(self.k)
        return Trip(J_new,pi_new,k_new)

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
