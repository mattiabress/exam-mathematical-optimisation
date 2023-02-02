from Point import Point
from Trip import Trip
import numpy as np
class Trips:
    # Swap two relocation moves between two taxi trips.
    @staticmethod
    def swap_random_realocation_moves(trip1,trip2):
        j1 = np.random.choice(trip1.J, 1)[0]
        j2 = np.random.choice(trip2.J, 1)[0]
        trip1.J.append(j2)
        trip2.J.append(j1)
        trip1.J.remove(j1)
        trip2.J.remove(j2)

    # Move a relocation move to another taxi trip
    @staticmethod
    def move_random_realocation_moves(current_trip, destination_trip):
        j1 = np.random.choice(current_trip.J, 1)[0]
        destination_trip.J.append(j1)
        current_trip.J.remove(j1)

    # Remove a drop-off point from a taxi trip
    @staticmethod
    def remove_random_dropoff(trip):
        pi = np.random.choice(trip.pi, 1)[0] # TODO controllare che non tolga inizio e fine che sono i due depositi
        trip.pi.remove(pi)

    # Swap two drop-off points between two taxi trips
    @staticmethod
    def swap_random_dropoff(trip1,trip2):
        pi1 = np.random.choice(trip1.pi, 1)[0]  # TODO controllare che non tolga inizio e fine che sono i due depositi
        pi2 = np.random.choice(trip2.pi, 1)[0]  # TODO controllare che non tolga inizio e fine che sono i due depositi
        trip1.pi.append(pi2)
        trip2.pi.append(pi1)
        trip1.pi.remove(pi1)
        trip2.pi.remove(pi2)

    # Add a drop-off point to a taxi trip
    @staticmethod
    def add_drop_off(trip,pi):
        trip.pi.append(pi) # TODO controllare che non tolga inizio e fine che sono i due depositi
        trip.k+=1

        # Add a drop-off point to a taxi trip

    @staticmethod
    def get_total_duration(trips):
        duration=0
        for trip in trips:
            duration+=trip.duration() # TODO sistemare
        return duration
