import numpy as np
from scipy.optimize import fsolve
from enzyme_kinetics_GluDH import kinetics, kinetic_params


class CSTR1Reactor:
    #decision vector x = [v_PPO, v_NAD, v_GluDH, v_FDH]
    v_names = ["PPO [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]"]
    LOWER = np.array([0.0001, 0.0005, 0.0001, 0.0001])
    UPPER = np.array([0.95, 0.05, 0.20, 0.20])

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    1000,
            "c_AFS":     20000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  180,
            "c_FDHS":  70.5,

            "X_PPO_target": 0.999,         #target conversion
            "tau_lo":        1.0,          #smallest residence time of the scan [min]
            "n_per_decade":  20,           #support points per decade of the continuation scan
            "max_decades":   12,           #largest residence time = tau_lo * 10**max_decades
            "tau_tol":       1e-6,         #relative accuracy of the residence time
        }
        self.ppo_final_history = []
        self.weights = np.array([1+self.params["c_PPOS"]/self.params["c_AFS"], 1.0, 1.0, 1.0]) #weights for optimized vector to acount for AF volume

    #Steady state balances==============================================================================================================================================================
    #y = c_in = [PPO, PPT, NADH, NAD, AF, E_GDH, E_FDH]
    def balances(self, y, tau, c_in):
        r1, r2, r3, r4, r5, r6 = kinetics(y)
        PPO, PPT, NADH, NAD, AF, E_GDH, E_FDH = y

        res = np.empty(7)
        res[0] = (c_in[0] - PPO)   / tau - r1
        res[1] = (c_in[1] - PPT)   / tau + r1
        res[2] = (c_in[2] - NADH)  / tau - r1 + r2 - r6
        res[3] = (c_in[3] - NAD)   / tau + r1 - r2 - r5
        res[4] = (c_in[4] - AF)    / tau - r2
        res[5] = (c_in[5] - E_GDH) / tau
        res[6] = (c_in[6] - E_FDH) / tau
        return res

    def _initial_guess(self, c_in):
        y0 = np.asarray(c_in, dtype=float).copy()
        if y0[2] <= 0.0:
            y0[2] = kinetic_params["Km_NADH"]     #Initial guess for NADH
        return y0

    def solve_cstr(self, tau, c_in, y_guess=None):
        if y_guess is None:
            y_guess = self._initial_guess(c_in)
        y_ss, info, ier, _ = fsolve(self.balances, y_guess, args=(tau, c_in), full_output=True)
        if ier != 1:
            #second attempt from the feed composition instead of the continuation guess
            y_alt, info_alt, _, _ = fsolve(self.balances, self._initial_guess(c_in),
                                           args=(tau, c_in), full_output=True)
            if np.max(np.abs(info_alt["fvec"])) < np.max(np.abs(info["fvec"])):
                y_ss = y_alt
        return np.maximum(y_ss, 0.0)

    #Residence time from target conversion===========================================================================================================================================
    def find_tau(self, c_in, X_target):
        p = self.params
        step = 10.0 ** (1.0 / p["n_per_decade"])

        tau = float(p["tau_lo"])
        y_guess = self._initial_guess(c_in)
        tau_lo = y_lo = None

        for _ in range(p["n_per_decade"] * p["max_decades"]):
            y_ss = self.solve_cstr(tau, c_in, y_guess)
            if 1.0 - y_ss[0] / c_in[0] >= X_target:
                break
            tau_lo, y_lo, y_guess = tau, y_ss, y_ss
            tau *= step
        else:
            raise ValueError(f"X = {X_target:.4f} not reachable up to tau = {tau:.3g} min")

        if tau_lo is None:
            return tau, y_ss

        #bisection in log space between tau_lo (X < X_target) and tau_hi (X >= X_target)
        tau_hi, y_hi = tau, y_ss
        while tau_hi / tau_lo > 1.0 + p["tau_tol"]:
            tau_mid = np.sqrt(tau_lo * tau_hi)
            y_mid = self.solve_cstr(tau_mid, c_in, y_lo)
            if 1.0 - y_mid[0] / c_in[0] >= X_target:
                tau_hi, y_hi = tau_mid, y_mid
            else:
                tau_lo, y_lo = tau_mid, y_mid

        return float(tau_hi), y_hi

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        v_PPO, v_NAD, v_GluDH, v_FDH = x

        #Punishment if total volume is unrealistic
        if (v_PPO +(p["c_PPOS"]/p["c_AFS"])*v_PPO+ v_NAD + v_GluDH + v_FDH) > 1.0:
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        #Feed concentrations from the feed volume fractions
        c_in = np.array([
            v_PPO * p["c_PPOS"],            #PPO
            0.0,                            #PPT
            0.0,                            #NADH
            v_NAD * p["c_NADS"],            #NAD
            v_PPO * p["c_PPOS"]/p["c_AFS"], #AF
            v_GluDH * p["c_GluDHS"],        #E_GDH
            v_FDH * p["c_FDHS"],            #E_FDH
        ])

        try:
            tau, y_ss = self.find_tau(c_in, p["X_PPO_target"])
        except Exception:
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        PPO = y_ss[0]
        PPT = y_ss[1]
        self.ppo_final_history.append(PPO)

        #objective functions
        sty     = PPT / (tau / 60)
        ton_e   = PPT / (c_in[5] + c_in[6])
        ton_cof = PPT / (c_in[2] + c_in[3])

        #for constraint
        X_PPO = 1.0 - PPO / c_in[0]

        return np.array([sty, ton_e, ton_cof, X_PPO])
