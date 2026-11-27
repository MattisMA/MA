import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from enzyme_kinetics_GluDH import kinetics, kinetic_params


class Reactor:
    #decision vector x = [v_PPO, v_NAD, v_GluDH, v_FDH, tau2]
    v_names = ["PPO [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]", "tau2 [min]"]
    LOWER = np.array([0.0001, 0.0005, 0.0001, 0.0001, 1.0])
    UPPER = np.array([0.95, 0.05, 0.20, 0.20, 1200.0])
    TAU1_MIN, TAU1_MAX = 1.0, 1200.0   #search interval for tau1 [min]

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    1000,
            "c_AFS":     8000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  180,
            "c_FDHS":  70.5,

            "X_target": 0.999,         #target conversion
            "X1":       0.7,           #fixed conversion at the outlet of reactor 1
            "n_tau":    5,             #evaluation after n_tau residence times
        }
        self.ppo_final_history = []
        self.tau1_history = []
        self.weights = np.array([1+self.params["c_PPOS"]*(1-kinetic_params["f_inert"])/self.params["c_AFS"], 1.0, 1.0, 1.0, 0.0]) #weights for optimized vector to acount for AF volume

    #Material balances==============================================================================================================================================================
    #y = c_in = [P_akt, P_slow, I_inert, PPT, NADH, NAD, AF, E_GDH, E_FDH]
    def balances(self, t, y, tau, c_in):
        r1, r2, r3, r4, r5, r6, r7 = kinetics(y)
        D = 1.0 / tau

        dPakt_dt   = D * (c_in[0] - y[0]) - r1 - r7
        dPslow_dt  = D * (c_in[1] - y[1]) + r7
        dIinert_dt = D * (c_in[2] - y[2])
        dPPT_dt    = D * (c_in[3] - y[3]) + r1
        dNADH_dt   = D * (c_in[4] - y[4]) - r1 + r2 - r6
        dNAD_dt    = D * (c_in[5] - y[5]) + r1 - r2 - r5
        dAF_dt     = D * (c_in[6] - y[6]) - r2
        dE_GDH_dt  = D * (c_in[7] - y[7]) - r3
        dE_FDH_dt  = D * (c_in[8] - y[8]) - r4

        return [dPakt_dt, dPslow_dt, dIinert_dt, dPPT_dt, dNADH_dt, dNAD_dt, dAF_dt, dE_GDH_dt, dE_FDH_dt]

    #Steady state of one CSTR=======================================================================================================================================
    #reactor initially filled with y0, evaluation after n_tau residence times
    def cstr_steady(self, tau, c_in, y0):
        sol = solve_ivp(
            fun=lambda t, y: self.balances(t, y, tau, c_in),
            t_span=(0.0, self.params["n_tau"] * tau),
            y0=y0,
            method="BDF",
            rtol=1e-8,
            atol=1e-10,
        )
        if sol.status != 0:
            raise RuntimeError("solver failed")
        return sol.y[:, -1]

    #Residence time of reactor 1 for given conversion X1 (root finding in log10(tau1))===========================================================================
    def tau_for_conversion(self, X1, c_in, c_usable_in):
        def f(log_tau):
            y = self.cstr_steady(10**log_tau, c_in, c_in)
            return (1.0 - (y[0] + y[1]) / c_usable_in) - X1

        lo, hi = np.log10(self.TAU1_MIN), np.log10(self.TAU1_MAX)
        if f(lo) >= 0: return self.TAU1_MIN        #X1 already reached at TAU1_MIN
        if f(hi) <= 0: return self.TAU1_MAX        #X1 not reachable -> tau1 clamped to TAU1_MAX
        return 10**brentq(f, lo, hi, xtol=1e-6)

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        kp = kinetic_params
        v_PPO, v_NAD, v_GluDH, v_FDH, tau2 = x
        X1 = p["X1"]

        #Speciation of the fed PPO
        c_PPO_in = v_PPO * p["c_PPOS"]
        c_inert_in = kp["f_inert"] * c_PPO_in
        c_usable_in = c_PPO_in - c_inert_in
        c_Pakt_in = kp["f_Pakt0"] * c_usable_in

        #Feed concentrations from the feed volume fractions
        c_in = np.array([
            c_Pakt_in,                      #P_akt
            c_usable_in - c_Pakt_in,        #P_slow
            c_inert_in,                     #I_inert
            0.0,                            #PPT
            0.0,                            #NADH
            v_NAD * p["c_NADS"],            #NAD
            c_usable_in,                    #AF
            v_GluDH * p["c_GluDHS"],        #E_GDH
            v_FDH * p["c_FDHS"],            #E_FDH
        ])

        #reactor 1: tau1 from conversion X1, reactor 2: fed with outlet of reactor 1
        try:
            tau1 = self.tau_for_conversion(X1, c_in, c_usable_in)
            y_1 = self.cstr_steady(tau1, c_in, c_in)
            y_f = self.cstr_steady(tau2, y_1, y_1)

        #Punishment if the solver failed
        except RuntimeError:
            self.ppo_final_history.append(c_usable_in)
            self.tau1_history.append(np.nan)
            return np.array([0.0, 0.0, 0.0, 0.0])

        PPO = y_f[0] + y_f[1]
        PPT = y_f[3]
        self.ppo_final_history.append(PPO)
        self.tau1_history.append(tau1)

        #objective functions
        sty     = PPT / ((tau1 + tau2) / 60)
        ton_e   = PPT / (c_in[7] + c_in[8])
        ton_cof = PPT / (c_in[4] + c_in[5])
        X_PPO = 1.0 - PPO / c_usable_in

        return np.array([sty, ton_e, ton_cof, X_PPO])
