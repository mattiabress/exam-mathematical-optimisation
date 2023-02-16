import matplotlib.pyplot as plt
import numpy as np
from Point import Point
class Simulation:
    @staticmethod
    def get_simulation_number(number_simulation):
        if number_simulation==0: # small
            n=5
            m=4
            ks=25
            kr=13
            kn=126
            T_start=2500
            c=0.126
            return (n,m,ks,kr,kn,T_start,c)
        elif number_simulation==1: # medium
            n = 30
            m = 8
            ks = 13
            kr = 25
            kn = 751
            T_start = 1251
            c = 0.126
            return (n,m,ks,kr,kn,T_start,c)
        else: # large
            n = 60
            m = 8
            ks = 25
            kr = 25
            kn = 751
            T_start = 1251
            c = 0.126
            return (n,m,ks,kr,kn,T_start,c)

    @staticmethod
    def plot_map(realocation_moves,drop_off_points, trips, figsize=(11, 14)):
        plt.style.use('_mpl-gallery')
        realocation_moves_x = []
        realocation_moves_y = []
        for point in realocation_moves:
            realocation_moves_x.append(point.u)
            realocation_moves_y.append(point.v)

        drop_off_points_x = []
        drop_off_points_y = []
        for drop_off in drop_off_points:
            drop_off_points_x.append(drop_off.u)
            drop_off_points_y.append(drop_off.v)

        # plot
        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        ax.scatter(realocation_moves_x, realocation_moves_y, c="r", label='Realocation moves')
        ax.scatter(drop_off_points_x, drop_off_points_y, c="b", label='Drop off points')
        # plot trips
        for idx, trip in enumerate(trips):
            trip_x = []
            trip_y = []
            for drop_off in trip.pi:
                trip_x.append(drop_off.u)
                trip_y.append(drop_off.v)
            ax.plot(trip_x, trip_y, '-', label=f'Trip {idx}')
            for i in range(len(trip_x)-1):
                ax.arrow(trip_x[i], trip_y[i], trip_x[i+1] - trip_x[i], trip_y[i+1] - trip_y[i],head_width=1, length_includes_head=True)

            color = ax.get_lines()[idx].get_color()
            for realocation_move in trip.J:
                ax.scatter(x=realocation_move.u, y=realocation_move.v, c=color)

        ax.set(xlim=(0, 110),
               ylim=(0, 140))

        # Major ticks every 20, minor ticks every 5
        major_ticks_x = np.arange(0, 110, 20)
        minor_ticks_x = np.arange(0, 110, 5)
        major_ticks_y = np.arange(0, 140, 20)
        minor_ticks_y = np.arange(0, 140, 5)

        ax.set_xticks(major_ticks_x)
        ax.set_xticks(minor_ticks_x, minor=True)
        ax.set_yticks(major_ticks_y)
        ax.set_yticks(minor_ticks_y, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        ax.legend()
        plt.show()

    @staticmethod
    def initialize_map(n_realocation_moves):
        J = []
        D = []
        for i in range(n_realocation_moves):
            x_v = np.random.randint(0, 20)  # random between 0 and 19
            v0 = 7.125
            if x_v == 0:
                v = v0
            elif x_v == 1:
                v = v0 + 4.25
            else:
                v = v0 + int(x_v % 2) * 4.25 + int(x_v / 2) * 5 + int(x_v / 2) * 8.50
            x_u = np.random.randint(0, 25)
            u0 = 7
            u = u0 + x_u * 4

            J.append(Point(u, v))
            delta = 4.625
            delta_2 = 8.875

            v_d = v - delta if int(x_v % 2) == 0 else v + delta
            v_d_4 = v + delta_2 if int(x_v % 2) == 0 else v - delta_2

            D.append(Point(2.5, v_d))  # D1 (2.5,v_d)
            D.append(Point(u, v_d))  # D2 (u,v_d)
            D.append(Point(107.5, v_d))
            D.append(Point(u, v_d_4))
        return (J, D)
