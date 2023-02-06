from Point import Point
from Trip import Trip
import numpy as np
class Trips:
    # Swap two relocation moves between two taxi trips.
    @staticmethod
    def swap_random_realocation_moves(trip1,trip2):
        if trip1.J==[] or trip2.J==[]: # list empty
            return

        j1 = np.random.choice(trip1.J, 1)[0]
        j2 = np.random.choice(trip2.J, 1)[0]
        trip1.J.append(j2)
        trip2.J.append(j1)
        trip1.J.remove(j1)
        trip2.J.remove(j2)


    # Move a relocation move to another taxi trip
    @staticmethod
    def move_random_realocation_moves(current_trip, destination_trip):
        if current_trip.J==[]:
            return
        j1 = np.random.choice(current_trip.J, 1)[0]
        destination_trip.J.append(j1)
        current_trip.J.remove(j1)

    # Remove a drop-off point from a taxi trip
    @staticmethod
    def remove_random_dropoff(trip):
        if trip.pi==[]:
            return
        idx = np.random.choice(range(len(trip.pi)), 1)[0]
        if idx == 0 or idx==len(trip.pi)-1:
            return
        trip.pi.remove(trip.pi[idx])

    # Swap two drop-off points between two taxi trips
    @staticmethod
    def swap_random_dropoff(trip1,trip2):
        if trip1.pi==[] or trip2.pi==[]:
            return
        #pi1 = np.random.choice(trip1.pi, 1)[0]  # TODO controllare che non tolga inizio e fine che sono i due depositi
        idx=np.random.choice(range(len(trip1.pi)), 1)[0]
        idx2 = np.random.choice(range(len(trip2.pi)), 1)[0]
        if idx==0 or idx==len(trip1.pi) or idx2==0 or idx==len(trip2.pi):
            return
        trip1.pi.append(trip2.pi[idx2])
        trip2.pi.append(trip1.pi[idx])
        trip1.pi.remove(trip1.pi[idx])
        trip2.pi.remove(trip2.pi[idx2])

    # Add a drop-off point to a taxi trip
    @staticmethod
    def add_drop_off(trip,pi):
        trip.pi.insert(len(trip.pi)-1,pi) # TODO controllare che non tolga inizio e fine che sono i due depositi
        trip.k+=1

        # Add a drop-off point to a taxi trip

    @staticmethod
    def get_total_duration(trips):
        duration=0
        for trip in trips:
            duration+=trip.trip_duration() # TODO sistemare
        return duration
