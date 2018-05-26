#!/usr/bin/env python3

from model_bank import ModelBank


class MMAC:
    """
    Multiple model adaptive control
    """

    STARTING_BLOOD_PRESSURE = 100.0
    DESIRED_BLOOD_PRESSURE_MIDDLE_STEP = 75.0
    DESIRED_BLOOD_PRESSURE = 50.0       # P_d -- pozadovany tlak pacienta
    MINIMAL_BLOOD_PRESSURE = DESIRED_BLOOD_PRESSURE - 20.0      # P_l -- minimalni tlak pacienta

    MAXIMAL_RECOMMENDED_DOSE = 600.0          # i_M -- 600 mikrogramu na kilo za hodinu, maximalni doporucena davka
    DRUG_CONCETRATION = 200.0          # C_s -- mikrogram/mililitr, koncentrace latky

    V = 0.05             # parameter controlling the convergence rate of W_j' with R_j,

    PATIENT_WEIGHT = 80.0     # W_p -- vaha pacienta v kilogramech

    def __init__(self):
        W_p = self.PATIENT_WEIGHT
        i_M = self.MAXIMAL_RECOMMENDED_DOSE
        C_s = self.DRUG_CONCETRATION
        self.U_M = W_p * i_M * (1.0 / C_s)         # Maximalni povolene davkovani

        self.P_l = self.MINIMAL_BLOOD_PRESSURE
        self.P_a = self.STARTING_BLOOD_PRESSURE      # pacientuv aktualni tlak

        self.model_bank = ModelBank()

    def limit_actual_infusion_rate(self, u_d):
        """ omezi infuzi do intervalu <0, U_M>, tak jak je spefikovano v rovnici 2, pro stanoveni "u"
        :rtype: return u
        """
        if u_d < 0.0:
            return 0.0
        elif u_d <= self.U_M:
            return u_d
        else:
            return self.U_M

    def low_pressure_check(self, u_c):
        """ Turn off drug infusion rate if patients blood pressure drops too low

        :param u_c: infusion rate before low blood pressure check
        :return: return 0 if patients blood pressure is too low, u_c otherwise
        """
        return u_c if self.P_a >= self.P_l else 0


if __name__ == "__main__":
    snt = MMAC()
