
class DSU:
    """Disjoint Set Union data structure.

    See more at: https://cp-algorithms.com/data_structures/disjoint_set_union.html
    """

    def __init__(self, n_elements):
        self.n = n_elements
        self.parent = list(range(n_elements))

    def new_set(self):
        """Insert a new set with index self.n."""
        self.parent.append(self.n)
        self.n += 1
        return self.n

    def find_root(self, x):
        """Get the root of the set, which contains x."""
        par = self.parent[x]
        if par == x:
            return x
        par = self.find_root(par)
        self.parent[x] = par
        return par

    def union_sets(self, x, y):
        """Merge the sets of x and y, return the root of the union."""
        x = self.find_root(x)
        y = self.find_root(y)
        if x != y:
            if x < y:
                x, y = y, x
            self.parent[y] = x
        return x
