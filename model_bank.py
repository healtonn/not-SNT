#!/usr/bin/env python3
import math

import numpy as np

from patient import Patient


class Model_bank:
    """
    Model banka, obsahujici vsech 7 modelu paciantu
    """
    MODEL_GAIN = [0.33, 0.62, 1.15, 2.10, 3.69, 6.06, 9.03]
    INFUSION_DELAY = [50, 50, 50, 50, 50, 50, 50]
    V = 0.05    # dle clanku

    def __init__(self, start_pressure, steps_in_1s, t):
        self.start_pressure = start_pressure
        self.t = t
        self.model_bank = [Patient(self.MODEL_GAIN[model_number], self.INFUSION_DELAY[model_number], start_pressure, steps_in_1s)
                           for model_number in range(len(self.MODEL_GAIN))]
        for patient in self.model_bank:
            patient.transfer_function_init(t)

        self.P_m = np.empty((len(self.t), len(self.model_bank)))
        self.R_j = np.empty_like(self.P_m)
        self.W_j1 = np.empty_like(self.P_m)
        self.W_num = np.zeros(len(self.MODEL_GAIN))

        for j in range(len(self.MODEL_GAIN)):
            self.P_m[0, j] = start_pressure
            self.R_j[0, j] = 0
            self.W_j1[0, j] = 1

    def sim_step(self, u, i, P_a, P_d):

        for j in range(len(self.MODEL_GAIN)):
            # simulacni krok vsech pacientu v model bank
            self.P_m[i, j] = self.model_bank[j].sim_step(u, i)
            # rovnice 8
            self.R_j[i, j] = (self.P_m[i, j] - P_a) / (self.start_pressure - P_d)
            # vypocet citatele v rovnice 5
            self.W_num[j] = (math.exp(math.pow(self.R_j[i, j], 2) *(-1) / 2 * math.pow(self.V, 2)) * self.W_j1[i-1, j])

        # vypocet jmenovatele v rovnice 5
        W_den = np.sum(self.W_num, dtype=np.double)
        for j in range(len(self.MODEL_GAIN)):
            # rovnice 5
            if(W_den != 0):
                self.W_j1[i, j] = self.W_num[j] / W_den
            else:
                self.W_j1[i, j] = self.W_num[j]