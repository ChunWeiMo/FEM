from src.simulation_parameter import SimulationParameter
from src.material_property import MaterialProperty
from src.mesh import Mesh

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
            i += 1

    @staticmethod
    def Nuemann(sim_para: SimulationParameter, mtrl_prop: MaterialProperty, mesh: Mesh):
        mesh.q[mesh.nodes-1] = -1e6

        mesh.dt[mesh.nodes-1] = 2 * mesh.F0
        mesh.dt[mesh.nodes-1] *= (mesh.t[mesh.nodes-2]-mesh.t[mesh.nodes-1])

        temp = 2 * sim_para.timestep * mesh.q[mesh.nodes-1]
        temp /= mtrl_prop.density * mtrl_prop.cp * mesh.dx
        mesh.dt[mesh.nodes-1] -= temp

        mesh.t[mesh.nodes-1] += mesh.dt[mesh.nodes - 1]

        i = 1
        while i <= sim_para.minimum_step or i <= sim_para.maximum_step:
            Scheme.explicit(sim_para, mtrl_prop, mesh)
            yield mesh.t
            i += 1
            print(f"iteration: {i}, {mesh.t}")
