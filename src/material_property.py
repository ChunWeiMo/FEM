class MaterialProperty():
    def __init__(self, name="", length=10, k=1e5, cp=1000, density_init=1000):
        self.name = name
        self.length = float(length)
        self.k = float(k)
        self.cp = float(cp)
        self.density = float(density_init)

    def __str__(self) -> str:
        return f"{self.name} property:\n length = {self.length}\n k = {self.k}"
