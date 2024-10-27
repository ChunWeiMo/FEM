class SimulationParameter():
    def __init__(self, timestep=1, minimum_step=10, maximum_step=100) -> None:
        self.timestep = float(timestep)
        self.minimum_step = int(minimum_step)
        self.maximum_step = int(maximum_step)
