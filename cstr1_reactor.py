import numpy as np
from scipy.optimize import least_squares
from enzyme_kinetics_GluDH import kinetics, kinetic_params


class Reactor:
    #decision vector x = [v_PPO, v_NAD, v_GluDH, v_FDH]
    v_names = ["PPO [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]", "tau [min]"]
    LOWER = np.array([0.0001, 0.0005, 0.0001, 0.0001, 1.0])
    UPPER = np.array([0.95, 0.05, 0.20, 0.20, 1200.0])

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    1000,
            "c_AFS":     20000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  180,
            "c_FDHS":  70.5,

            "X_target": 0.999,         #target conversion
        }
        self.ppo_final_history = []
        self.weights = np.array([1+self.params["c_PPOS"]/self.params["c_AFS"], 1.0, 1.0, 1.0, 0.0]) #weights for optimized vector to acount for AF volume

    #Steady state balances==============================================================================================================================================================
    #y = c_in = [PPO, PPT, NADH, NAD, AF, E_GDH, E_FDH]
    def balances(self, y, tau, c_in):
        r1, r2, r3, r4, r5, r6= kinetics(y)
        c = np.array([
            c_in[0] - r1 * tau, #PPO
            c_in[1] + r1 * tau, #PPT
            c_in[2] + (-r1+r2-r6) * tau, #NADH
            c_in[3] + (r1-r2-r5) * tau, #NAD
            c_in[4] - r2 *tau,  #AF
            c_in[5],            #E_GluDH
            c_in[6],            #E_FDH 

        ])
        return y - c

    def solve_cstr(self, tau, c_in, y_guess=None):
        c_in = np.asarray(c_in, dtype=float)
        Cof_in = c_in[2] + c_in[3]

        scale = np.maximum(np.array([c_in[0], c_in[0], Cof_in, Cof_in, c_in[4], c_in[5], c_in[6]]), 1e-12)
        
        upper = np.maximum(np.array([c_in[0], c_in[1] + c_in[0], Cof_in, Cof_in, c_in[4], np.inf, np.inf]), 1e-12)
        
        if y_guess is None or not np.all(np.isfinite(y_guess)):
            y_guess = c_in.copy()
            y_guess[2] = min(kinetic_params["Km_NADH"], Cof_in)
            y_guess[3] = Cof_in - y_guess[2]
        y0 = np.clip(np.asarray(y_guess, dtype=float), 0.0, upper)

        sol = least_squares(lambda y: self.balances(y, tau, c_in) / scale,
                             y0,
                             bounds=(np.zeros(7),
                             upper), x_scale="jac",
                             xtol=1e-12,
                             ftol=1e-12)

        if np.max(np.abs(sol.fun)) > 1e-9:
            return np.full(7, np.nan)
        return np.maximum(sol.x, 0.0)

        return np.maximum(y_ss, 0.0)

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        v_PPO, v_NAD, v_GluDH, v_FDH, tau = x

        #Punishment if total volume is unrealistic
        if (v_PPO +(p["c_PPOS"]/p["c_AFS"])*v_PPO+ v_NAD + v_GluDH + v_FDH) > 1.0:
            self.ppo_final_history.append(np.nan)
            return np.array([0.0, 0.0, 0.0, 0.0])
        

        #Feed concentrations from the feed volume fractions
        c_in = np.array([
            v_PPO * p["c_PPOS"],            #PPO
            0.0,                            #PPT
            0.0,                            #NADH
            v_NAD * p["c_NADS"],            #NAD
            v_PPO * p["c_PPOS"],            #AF
            v_GluDH * p["c_GluDHS"],        #E_GDH
            v_FDH * p["c_FDHS"],            #E_FDH
        ])

        y_ss = self.solve_cstr(tau, c_in)

        #Punishment if the steady state solver did not converge
        if np.max(np.abs(self.balances(y_ss, tau, c_in))) > 1e-6 * c_in[0] / tau:
            self.ppo_final_history.append(c_in[0])
            return np.array([0.0, 0.0, 0.0, 0.0])

        PPO = y_ss[0]
        PPT = y_ss[1]
        self.ppo_final_history.append(PPO)

        #objective functions
        sty     = PPT / (tau / 60)
        ton_e   = PPT / (c_in[5] + c_in[6])
        ton_cof = PPT / (c_in[2] + c_in[3])
        X_PPO = 1.0 - PPO / c_in[0]

        return np.array([sty, ton_e, ton_cof, X_PPO])
