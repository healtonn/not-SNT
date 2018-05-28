#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from model_bank import ModelBank
from patient import Patient
from pid_controller import Pid


# class pid(object):
#     # PID tuning
#     Kc = 2.0
#     tauI = 10.0
#     tauD = 0
#     sp = []         # set point

class MMAC():
    """
    Multiple model adaptive control
    """

    STARTING_BLOOD_PRESSURE = 100.0
    DESIRED_BLOOD_PRESSURE_MIDDLE_STEP = 75.0
    DESIRED_BLOOD_PRESSURE = 50.0       # P_d -- pozadovany tlak pacienta
    MINIMAL_BLOOD_PRESSURE = DESIRED_BLOOD_PRESSURE - 20.0      # P_l -- minimalni tlak pacienta

    MAXIMAL_RECOMMENDED_DOSE = 60.0          # i_M -- 600 mikrogramu na kilo za sekundu, maximalni doporucena davka
    DRUG_CONCETRATION = 200.0          # C_s -- mikrogram/mililitr, koncentrace latky

    V = 0.05             # parameter controlling the convergence rate of W_j' with R_j,

    PATIENT_WEIGHT = 80.0     # W_p -- vaha pacienta v kilogramech

    def controll_loop(self, change_pressure_time):
        time_of_sim_s = 1200  # delka simulace v 1200s = 20min
        steps_in_1s = 1  # kolik kroku simulace se provede v jedne sekunde
        sim_steps = (time_of_sim_s + 1) * steps_in_1s  # pocet kroku simulace

        t = np.linspace(0, time_of_sim_s, sim_steps)  # jednotlive kroky simulace
        delta_t = t[1] - t[0]

        # Maximalni povolene davkovani. Musim prepocitat davku na zaklade kroku simulace
        self.U_M = self.PATIENT_WEIGHT * self.MAXIMAL_RECOMMENDED_DOSE / steps_in_1s * (1.0 / self.DRUG_CONCETRATION)

        # v desire_p je prubeh pozadovaneho tlaku
        change_desire_pressure_steps = change_pressure_time * steps_in_1s
        desire_p = np.full_like(t, self.DESIRED_BLOOD_PRESSURE_MIDDLE_STEP)
        desire_p[change_desire_pressure_steps: ] = self.DESIRED_BLOOD_PRESSURE

        p_a = self.STARTING_BLOOD_PRESSURE
        patient = Patient(2.1, 50, self.STARTING_BLOOD_PRESSURE, steps_in_1s)
        patient.transfer_function_init(t)

        # inicializuji pid controller jako PI (PID bez derivacni slozky)
        pid = Pid(2.0, 10.0, 0, desire_p)
        pid.init_pid(t, delta_t, self.U_M, self.STARTING_BLOOD_PRESSURE)

        for i in range(0, sim_steps-1):
            u_c = pid.sim_step(i)
            # block F2 je zakomponovan v pid a je predrazen blocku F1, coz nevadi
            # volam funckci blocku F1 ze chematu 2
            u = self.low_pressure_check(p_a, u_c, desire_p[i])
            if i >= 1:
                p_a = patient.sim_step(u, i)
            pid.set_new_pv(i, p_a)

        pid.op[sim_steps-1] = pid.op[sim_steps-2]   #oprava posledni velici
        # patient.plot_patient_pressure()
        pv = pid.pv
        op = pid.op
        self.plot_response(t, pv, op, desire_p)

    def plot_response(self, t, pv, op, sp):
        plt.figure(1)

        plt.subplot(2, 1, 1)
        plt.plot(t, sp, 'k-', linewidth=1, label='Set Point (SP)')
        plt.plot(t, pv, 'b--', linewidth=1, label='Process Variable (PV)')
        plt.legend(loc='best')
        plt.ylabel('Process Output')

        plt.subplot(2, 1, 2)
        plt.plot(t, op, 'r:', linewidth=1, label='Controller Output (OP)')
        plt.legend(loc='best')
        plt.ylabel('Process Input')

        plt.xlabel('Time')
        plt.show()

    def limit_actual_infusion_rate(self, u_d):
        """ omezi infuzi do intervalu <0, U_M>, tak jak je spefikovano v rovnici 2, pro stanoveni "u"
        :return: return u
        """
        if u_d < 0.0:
            return 0.0
        elif u_d <= self.U_M:
            return u_d
        else:
            return self.U_M

    def low_pressure_check(self, P_a, u_c, P_d):
        """ Turn off drug infusion rate if patients blood pressure drops too low
        :param P_a: actual patient pressure
        :param u_c: infusion rate before low blood pressure check
        :param P_d: desired pressure.
        :return: return 0 if patients blood pressure is too low, u_c otherwise
        """
        return u_c if P_a >= P_d - 20 else 0


if __name__ == "__main__":
    model = MMAC()
    model.controll_loop(600)