import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


class Patient:
    """
    Tato trida modeluje pacienta, podle obrazku 3
    Hlavnimi parametry urcujicimi reakci pacienta na SNP (sodium nitroprusside)
    jsou plant_gain a infusion_delay
    """
    ALPHA = 0.5  # fraction of SNP recirculated
    tau1 = 50.0  # time constant of SNP action in seconds
    tau2 = 10.0  # time constant for flow through pulmonary circulation in seconds
    tau3 = 30.0  # time constant for flow through systemic circulation in seconds

    def __init__(self, plant_gain, infusion_delay, start_pressure, steps_in_1s):
        self.G = plant_gain
        self.infusion_delay = infusion_delay * steps_in_1s  # zpozdeni reakce na SNP (v krocich)
        self.start_pressure = start_pressure  # pocatecni tlak pacienta

    def transfer_function_init(self, time_points):
        """
        Vytvori model pacienta (sys) z transfer function v laplasove tvaru.
        Zaroven nachysta pole pro ukladani:
            time-evolution vector (x)
            drop in pressure due to SNP (y)
            patient pressure (p)
        :param time_points: pole casovych kroku
        """
        # den - jmenovatel v polynomialnim tvaru
        # num - citatel v polynomialnim tvaru
        den_s3 = self.tau3 * self.tau2 * self.tau1
        den_s2 = self.tau3 * self.tau2 + self.tau3 * self.tau1 + self.tau2 * self.tau1
        den_s1 = self.tau3 + self.tau2 + self.tau1 - self.ALPHA * self.tau2
        den_s0 = 1 - self.ALPHA
        num = np.array([self.G * self.tau3, self.G])
        den = np.array([den_s3, den_s2, den_s1, den_s0])
        self.sys = signal.TransferFunction(num, den)

        self.time_points = time_points

        # pole pro ukladani time-evolution vector (self.x)
        self.x = np.empty((len(time_points) + self.infusion_delay, 3))
        # na prvnich infusion_delay_in_steps pozic nastavim za x 0
        self.x[:self.infusion_delay + 1, :] = 0

        # pole pro ukladani "drop in pressure due to SNP" (self.y)
        self.y = np.empty(len(time_points) + self.infusion_delay)
        # kvuli zpozdeni na prvnich infusion_delay_in_steps pozic nastavim start_pressure
        self.y[:self.infusion_delay + 1] = 0

        # prubeh tlaku pacienta
        self.p = np.empty(len(time_points) + self.infusion_delay)
        self.p[:self.infusion_delay + 1] = self.start_pressure

    def sim_step(self, dose, sim_step):
        """
        provede krok simulace krevniho tlaku pacienta
        do instancni promenne y uklada chovani tlaku pacienta
        :param dose: mnozstvi davky podane pacientovi v kroku sim_step
        :param sim_step: zacina od 1
        """
        tspan = [self.time_points[sim_step - 1], self.time_points[sim_step]]
        dose2 = [dose, dose]
        tout, yout, xout = signal.lsim2(self.sys, dose2, tspan, self.x[sim_step + self.infusion_delay - 1])
        self.y[sim_step + self.infusion_delay] = yout[1]
        self.p[sim_step + self.infusion_delay] = self.start_pressure - yout[1]
        self.x[sim_step + self.infusion_delay] = xout[1]

    def get_pressure(self, sim_step):
        """
        :param sim_step: krok v kterem chceme znat tlak pacienta
        :return: tlak pacienta v kroku sim_step
        """
        return self.p[sim_step + self.infusion_delay]

    def plot_patient_pressure(self):
        plt.figure(1)

        plt.subplot(2, 1, 1)
        y_cut = self.y[:len(self.time_points)]
        plt.plot(self.time_points, y_cut, 'r-')
        plt.title('drop in pressure due to SNP')

        plt.subplot(2, 1, 2)
        p_cut = self.p[:len(self.time_points)]
        plt.plot(self.time_points, p_cut, 'r-')
        plt.title('patient pressure')

        plt.show()


if __name__ == "__main__":
    time_of_sim_s = 1200            # delka simulace v 1200s = 20min
    steps_in_1s = 1                 # kolik kroku simulace s provede v jedne sekunde
    sim_steps = (time_of_sim_s + 1) * steps_in_1s   # pocet kroku simulace

    t = np.linspace(0, time_of_sim_s, sim_steps)    # jednotlive kroky simulace

    u = np.zeros(sim_steps)         # pro testovani. vektor hodnot SNP
    u[100:500] = 5.0
    u[500:800] = 10.0
    u[801:] = 0.0

    patient = Patient(2.1, 50, 100, steps_in_1s)
    patient.transfer_function_init(t)
    for i in range(1, sim_steps):
        patient.sim_step(u[i], i)
        # patient.get_pressure(i)      #vrati tlak pacienta v kroku i

    patient.plot_patient_pressure()
