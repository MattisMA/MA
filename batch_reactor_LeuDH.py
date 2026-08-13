import numpy as np
from scipy.integrate import solve_ivp
from enzyme_kinetics_LeuDH import kinetics


class Reactor:
    """Batch reactor simulation: material balances + reaction simulation."""

    #optimized vector x = [v_TMP0, v_NAD0, v_LeuDH, v_FDH]
    v_names = ["TMP [ml/ml]", "NAD [ml/nl]", "E_LeuDH [ml/ml]", "E_FDH [ml/ml]"]
    LOWER = np.array([0.0001, 0.0005, 0.0001, 0.0001])
    UPPER = np.array([0.95, 0.05, 0.20, 0.20])

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_TMPS":    1000,
            "c_ForS":     20000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_LeuDHS":  68.493,
            "c_FDHS":  48.0,

            "X_target": 0.999,         #target conversion
            "T_max": 1200.0,               #max reaction time [min]
        }
        self.ppo_final_history = []
        self.weights = np.array([1+self.params["c_TMPS"]/self.params["c_ForS"], 1.0, 1.0, 1.0]) #weights for optimized vector to acount for AF volume

    #Material balances==============================================================================================================================================================
    def balances(self, t, y):
        r1, r2, r3, r4, r5, r6 = kinetics(y)

        #Material balances
        dTMP_dt   = -r1 #- r7
        dTLeu_dt   = +r1
        dNADH_dt  = -r1 + r2 - r6
        dNAD_dt   = +r1 - r2 - r5
        dFor_dt    = -r2
        dE_LeuDH_dt = -r3
        dE_FDH_dt = -r4
        #dPPX_dt = r7

        return [dTMP_dt, dTLeu_dt, dNADH_dt, dNAD_dt, dFor_dt, dE_LeuDH_dt, dE_FDH_dt]

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        v_TMP0, v_NAD0, v_LeuDH, v_FDH = x

        #Substrate/Enzyme concentrations from starting volume
        c_TMP0 = v_TMP0 * p["c_TMPS"]
        c_TLeu0 = 0.0
        c_For0 = c_TMP0
        c_NADH0 = 0.0
        c_NAD0 = v_NAD0 * p["c_NADS"]
        LeuDH = v_LeuDH * p["c_LeuDHS"]
        FDH  = v_FDH * p["c_FDHS"]

        y0 = [c_TMP0, c_TLeu0, c_NADH0, c_NAD0, c_For0, LeuDH, FDH]

        #Punishment if total volume is unrealistic
        if (v_TMP0+(p["c_TMPS"]/p["c_ForS"])*v_TMP0+v_NAD0+v_LeuDH+v_FDH) > 1.0:
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        #event: target conversion is reached
        def event_X99(t, y):
            return y[0] - y0[0] * (1 - p["X_target"])
        event_X99.terminal = False
        event_X99.direction = -1

        sol = solve_ivp(
            fun=self.balances,
            t_span=(0.0, p["T_max"]),
            y0=y0,
            method="BDF",
            events=event_X99,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10,
        )

        #solving for reaction time and TMP and TLeu concentrations when X_target is reached
        hit = sol.t_events[0].size > 0
        tf  = float(sol.t_events[0][0]) if hit else sol.t[-1]
        y_f = sol.sol(tf) if hit else sol.y[:, -1]
        TMP = y_f[0]
        TLeu = y_f[1]

        #final TMP concentration after T_max
        TMP_final = sol.y[0, -1]

        self.ppo_final_history.append(TMP)

        #objective functions
        sty     = TLeu / (tf / 60)
        ton_e   = TLeu / (LeuDH + FDH)
        ton_cof = TLeu / (c_NADH0 + c_NAD0)

        #for constraint: X_TMP after T_max
        X_TMP = 1.0 - TMP_final/c_TMP0

        return np.array([sty, ton_e, ton_cof, X_TMP])
