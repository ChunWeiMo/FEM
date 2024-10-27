import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class SimulationParameter():
    def __init__(self, timestep=1, minimum_step=10, maximum_step=100) -> None:
        self.timestep = float(timestep)
        self.minimum_step = int(minimum_step)
        self.maximum_step = int(maximum_step)


class MaterialProperty():
    def __init__(self, name="", length=10, k=1e5):
        self.name = name
        self.length = float(length)
        self.k = float(k)

    def __str__(self) -> str:
        return f"{self.name} property:\n length = {self.length}\n k = {self.k}"


class Mesh():
    def __init__(self, nodes=10, length=0, cp=1000, t_init=1000, density_init=1000):
        self.nodes = int(nodes)
        self.index = np.array(np.linspace(1, nodes, nodes))
        self.t = np.full(nodes, t_init, dtype=float)
        self.dt = np.zeros(nodes, dtype=float)
        self.density = np.full(nodes, density_init, dtype=float)
        self.dx = float(length) / float(nodes)
        self.cp = np.full(nodes, fill_value=cp, dtype=float)
        self.res = np.full(nodes, fill_value=1, dtype=float)

    def __str__(self) -> str:
        msg = f"Temperature: {self.t}, {len(self.t)} nodes\n"
        return msg


class Scheme():
    @staticmethod
    def explicit(sim_para: SimulationParameter, mtrl_prop: MaterialProperty, mesh: Mesh):
        for n in range(1, mesh.nodes-1):
            mesh.dt[n] = mtrl_prop.k * sim_para.timestep
            mesh.dt[n] /= (mesh.dx**2.0) * mesh.density[n] * mesh.cp[n]
            mesh.dt[n] *= mesh.t[n-1]-2 * mesh.t[n]+mesh.t[n+1]
            mesh.t[n] += mesh.dt[n]
            # print(f"n: {n}, mesh.dt[n]: {mesh.dt[n]}, mesh.t[n]: {mesh.t[n]}, type{type(mesh.t[n])}")

    @staticmethod
    def Dirichlet(sim_para: SimulationParameter, mtrl_prop: MaterialProperty, mesh: Mesh):
        mesh.t[0] = 1000
        mesh.t[mesh.nodes-1] = 1100

        i = 1
        while i <= sim_para.minimum_step or i <= sim_para.maximum_step:
            Scheme.explicit(sim_para, mtrl_prop, mesh)
            yield mesh.t
            print(f"iteration: {i} {mesh}")
            i += 1


def run_simulation(sim_para: SimulationParameter, mtrl_prop: MaterialProperty, mesh: Mesh):
    fig, ax = plt.subplots()
    line, = ax.plot(mesh.index, mesh.t, marker="*")
    ax.set_ylim(900, 1300)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature distribution")

    def update_animation(temperature):
        line.set_ydata(temperature)
        return line
    
    ani = FuncAnimation(fig, update_animation, frames=Scheme.Dirichlet(
        sim_para, mtrl_prop, mesh), interval=42)
    plt.show()


def main():
    metal_plate = MaterialProperty(name="metal_plate", length=10, k=150000)
    print(metal_plate)
    mesh = Mesh(nodes=10, length=metal_plate.length, t_init=1000.0)
    sim_para = SimulationParameter(timestep=1, maximum_step=200)
    
    run_simulation(sim_para=sim_para, mtrl_prop=metal_plate, mesh=mesh)


if __name__ == "__main__":
    main()
