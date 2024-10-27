import numpy as np


class Mesh():
    def __init__(self, nodes=10, length=0, cp=1000, t_init=1000, density_init=1000):
        self.nodes = int(nodes)
        self.index = np.array(np.linspace(1, nodes, nodes))
        self.t = np.full(nodes, t_init, dtype=float)
        self.dt = np.zeros(nodes, dtype=float)
        self.density = np.full(nodes, density_init, dtype=float)
        self.dx = float(length) / float(nodes)
        self.cp = np.full(nodes, fill_value=cp, dtype=float)
        self.F0 = float(0)
        self.q = np.zeros(nodes, dtype=float)
        self.res = np.full(nodes, fill_value=1, dtype=float)

    def __str__(self) -> str:
        msg = f"Temperature: {self.t}, {len(self.t)} nodes\n"
        return msg
