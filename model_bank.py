#!/usr/bin/env python3


class ModelBank:
    """
    Model banka, obsahujici vsech 7 modelu paciantu
    """
    MODEL_GAIN = [0.33, 0.62, 1.15, 2.10, 3.69, 6.06, 9.03]

    def __init__(self):
        self.model_bank = [Model(self.MODEL_GAIN[model_number], model_number) for model_number in range(len(self.MODEL_GAIN))]


class Model:
    # TODO WHITE GAUSSIAN NOISE - budeme asi potřebovat
    """
    Tato classa reprezentuje jeden model v model bance, podle obrazku 3
    """
    ALPHA = 0.5     # fraction of SNP recirculated
    t_1 = 50.0      # time constant of SNP action in seconds
    t_2 = 10.0      # time constant for flow through pulmonary circulation in seconds
    t_3 = 30.0      # time constant for flow through systemic circulation in seconds
    T = 50.0        # infusion delay in seconds

    def __init__(self, plant_gain, model_number):
        self.G = plant_gain
        self.delta = self.set_delta(model_number)
        print("startuju s parametrem ", self.G, " a deltou: ", self.delta)

    # delta se má nastavovat na základě toho jak je daný model dobrý, ne takto na základě čísla, viz začátek kapitoly 3
    @staticmethod
    def set_delta(model_number):
        if model_number <= 2:
            return 0.01
        elif model_number <= 4:
            return 0.002
        else:
            return 0.0004
