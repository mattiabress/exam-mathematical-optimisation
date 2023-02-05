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
