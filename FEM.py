import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from src.simulation_parameter import SimulationParameter
from src.material_property import MaterialProperty
from src.mesh import Mesh
from src.scheme import Scheme


def run_simulation(sim_para: SimulationParameter, mtrl_prop: MaterialProperty, mesh: Mesh):
    fig, ax = plt.subplots()
    line, = ax.plot(mesh.index, mesh.t, marker="*")
    ax.set_ylim(980, 1005)
    ax.set_xlabel("Nodes")
    ax.set_ylabel("Temperature")
    ax.set_title("Temperature distribution")

    def update_animation(temperature):
        line.set_ydata(temperature)
        return line

    ani = FuncAnimation(fig, update_animation, frames=Scheme.Nuemann(
        sim_para, mtrl_prop, mesh), interval=42, cache_frame_data=False)
    plt.show()


def main():
    metal_plate = MaterialProperty(name="metal_plate", length=10, k=150000)
    print(metal_plate)
    mesh = Mesh(nodes=10, length=metal_plate.length, t_init=1000.0, density_init=200)
    sim_para = SimulationParameter(timestep=1, maximum_step=20)
    mesh.F0 = metal_plate.k * sim_para.timestep
    mesh.F0 /= (mesh.dx**2.0) * metal_plate.density * metal_plate.cp

    run_simulation(sim_para=sim_para, mtrl_prop=metal_plate, mesh=mesh)


if __name__ == "__main__":
    main()
