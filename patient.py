import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class Patient:
    """
    Tato trida modeluje pacienta, podle obrazku 3
    """
    ALPHA = 0.5  # fraction of SNP recirculated
    tau1 = 50.0  # time constant of SNP action in seconds
    tau2 = 10.0  # time constant for flow through pulmonary circulation in seconds
    tau3 = 30.0  # time constant for flow through systemic circulation in seconds

    def __init__(self, plant_gain, infusion_delay):
        self.G = plant_gain
        self.infusionDelay = infusion_delay

    def transfer_function(self):
        den_s3 = self.tau3 * self.tau2 * self.tau1
        den_s2 = self.tau3 * self.tau2 + self.tau3 * self.tau1 + self.tau2 * self.tau1
        den_s1 = self.tau3 + self.tau2 + self.tau1 - self.ALPHA * self.tau2
        den_s0 = 1 - self.ALPHA
        num = np.array([self.G * self.tau3, self.G])
        den = np.array([den_s3, den_s2, den_s1, den_s0])
        sys = signal.TransferFunction(num, den)
        t, y = signal.step(sys)
        plt.figure(1)
        plt.plot(t, y, 'r-')
        plt.show()


if __name__ == "__main__":
    patient = Patient(8, 50)
    patient.transfer_function()
