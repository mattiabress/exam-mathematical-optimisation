class Point:
    def __init__(self, u, v):
        self.u = u
        self.v = v

    def __str__(self):
        return f'({self.u},{self.v})'

    def __copy__(self):
        return Point(self.u, self.v)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Point):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return self.u == other.u and self.v == other.v

    @staticmethod
    def remove_from_list(list_points, selected_point):
        new_list = []
        for point in list_points:
            if not selected_point == point:
                new_list.append(point)
        return new_list
