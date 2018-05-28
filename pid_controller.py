import numpy as np

class Pid:
    """
    PID controller, integracni a derivacni slozka se da odstranit dosazenim
    za tauI = 0, popripade tauD = 0
    """
    def __init__(self, Kc, tauI, tauD, desire_p):
        # vim ze by se to nemeli michat jazyky komentaru, ale obcas nevim jak to
        # cesky napsat a yaroven se mi stale pise lepsi v cestine
        """
        :param Kc: tunning parametr for proportional part of controller
        :param tauI: tunning parametr for integral  part of controller
        :param tauD: tunning parametr for derivative  part of controller
        :param desire_p: pozadovany tlak pacienta v 1D poli (hodonoty ve vsech casech)
        """
        self.tauD = tauD
        self.tauI = tauI
        self.Kc = Kc
        self.desire_p = desire_p

    def init_pid(self, time_points, delta_t, U_M, start_pressure):
        """
        Inicializace PID controlleru
        Ukladam vsechno do pole, i kdyz bych nemusel, abych si mohl prubehy
        jednotlivych parametru zobrazovat a lepe nastavovat regulator
        :param time_points: cesy v kterych se bude vyhodnocovat
        :param delta_t: velikost kroku metody
        :param U_M: maximalni mozna davka pro pacienta
        :param start_pressure: pocatecni tlak pacienta
        """
        self.delta_t = delta_t
        self.U_M = U_M
        self.op = np.zeros_like(time_points)    # controller output     u_c
        self.pv = np.zeros_like(time_points)    # process variable      pp patient pressure
        self.pv[0] = start_pressure
        self.e = np.zeros_like(time_points)     # rozdil pozadovaneho a skutecneho tlaku
        self.ie = np.zeros_like(time_points)    # integral of the error
        self.dpv = np.zeros_like(time_points)   # derivative of the pv
        self.P = np.zeros_like(time_points)     # proportional
        self.I = np.zeros_like(time_points)     # integral
        self.D = np.zeros_like(time_points)     # derivative

    def sim_step(self, i):
        self.e[i] = self.pv[i] - self.desire_p[i]

        if i >= 1:  # jelikoz potrebuji hodnoty z minuleho kroku muzu pocitat az od kroku 1
            self.dpv[i] = (self.pv[i] - self.pv[i - 1]) / self.delta_t
            self.ie[i] = self.ie[i - 1] + self.e[i] * self.delta_t

        self.P[i] = self.Kc * self.e[i]
        self.I[i] = self.Kc / self.tauI * self.ie[i]
        self.D[i] = - self.Kc * self.tauD * self.dpv[i]
        self.op[i] = self.op[0] + self.P[i] + self.I[i] + self.D[i]

        # implementace bloku F2 z obrazku 2 (rovnice 2)
        if self.op[i] >= self.U_M:
            self.op[i] = self.U_M
            self.ie[i] = self.ie[i] - self.e[i] * self.delta_t  # anti-reset windup
        if self.op[i] < 0:  # check lower limit
            self.op[i] = 0
            self.ie[i] = self.ie[i] - self.e[i] * self.delta_t  # anti-reset windup

        return self.op[i]

    def set_new_pv(self, i, pv):
        self.pv[i + 1] = pv
