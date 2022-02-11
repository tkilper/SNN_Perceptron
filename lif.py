from collections import namedtuple
from dataclasses import dataclass, asdict

# define the output format of this neuron
LIFRecord = namedtuple("LIFRecord", "v s i")


# define the input format of this neuron
@dataclass
class LIFParams:
    capacitance: float = 1.6
    resistance: float = 1.5
    equilibrium: float = -30
    threshold: float = -20


class LIF:
    def __init__(self, lif_params=LIFParams()):
        self.p = lif_params
        self.V = lif_params.equilibrium

    def set_params(self, lif_params):
        self.p = lif_params

    def get_params(self):
        return asdict(self.p)

    def reset(self):
        self.V = self.p.equilibrium

    def step(self, ext_i, dt, callback=None):
        s = int(self.V >= self.p.threshold)
        self.V = s * self.p.equilibrium + (1 - s) * (
                self.V
                - dt
                / (self.p.resistance * self.p.capacitance)
                * ((self.V - self.p.equilibrium) - ext_i * self.p.resistance)
        )

        # spike value is binary, alter 0s to hide them from the plot
        # todo change if using a plotting library that can exclude 0s
        rs = (1 - s) * -200 + s  # arbitrary out of plot boundaries integer

        r = LIFRecord(v=self.V, s=rs, i=ext_i)

        if callback:
            callback(r)

        return r