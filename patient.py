import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from transferfunctiondelay import TransferFunctionDelay, lsim3


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
        sys = TransferFunctionDelay(self.infusionDelay, num, den)
        # sys = signal.TransferFunction(num, den)

        lenght = 1000
        t = np.linspace(0, lenght-1, lenght)
        u1 = np.full(300, 20, dtype=np.double)
        u2 = np.full(200, 5, dtype=np.double)
        u3 = np.full(200, 25, dtype=np.double)
        u4 = np.full(300, 2, dtype=np.double)
        u = np.concatenate([u1, u2, u3, u4])
        # tout2, y2, x2 = signal.lsim2(sys, u, t)
        tout2, y2, x2 = lsim3(sys, u, t)
        t2 = np.linspace(lenght, 1500, 500)
        # x0 = x2[lenght-1]
        u1 = np.full(200, 25, dtype=np.double)
        u2 = np.full(300, 0, dtype=np.double)
        u = np.concatenate([u1, u2])
        # tout, y, x = signal.lsim2(sys, u, t2, x2[lenght-1])
        tout, y, x = lsim3(sys, u, t2, x2[lenght-1])

        y_final = np.concatenate([y2, y])
        t3 = np.linspace(0, 1500, 1500)
        plt.figure(1)
        plt.plot(t3, y_final, 'r-')
        plt.show()


if __name__ == "__main__":
    patient = Patient(8, 50)
    patient.transfer_function()
