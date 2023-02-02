class Point:
    def __init__(self, u, v):
        self.u = u
        self.v = v
    def __str__(self):
        return f'({self.u},{self.v})'

    def __copy__(self):
        return Point(self.u,self.v)

