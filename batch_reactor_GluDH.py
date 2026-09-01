import numpy as np
from scipy.integrate import solve_ivp
from enzyme_kinetics_GluDH import kinetics, kinetic_params


class Reactor:
    """Batch reactor simulation: material balances + reaction simulation."""

    #optimized vector x = [v_PPO0, v_NAD0, v_GluDH, v_FDH]
    v_names = ["PPO [ml/ml]", "NAD [ml/nl]", "E_GluDH [ml/ml]", "E_FDH [ml/ml]"]
    LOWER = np.array([0.0001, 0.0005, 0.0001, 0.0001])
    UPPER = np.array([0.95, 0.05, 0.20, 0.20])

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    1000,
            "c_AFS":     8000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  170,
            "c_FDHS":  110,

            "X_target": 0.999,         #target conversion
            "T_max": 1200.0,               #max reaction time [min]
        }
        self.ppo_final_history = []
        self.weights = np.array([1+self.params["c_PPOS"]/self.params["c_AFS"], 1.0, 1.0, 1.0]) #weights for optimized vector to acount for AF volume
    #Material balances==============================================================================================================================================================
    def balances(self, t, y):
        r1, r2, r3, r4, r5, r6, r7 = kinetics(y)

        #Material balances
        dPakt_dt  = -r1 - r7
        dPslow_dt = +r7
        dPinert_dt = 0
        dPPT_dt   = +r1
        dNADH_dt  = -r1 + r2 - r6
        dNAD_dt   = +r1 - r2 - r5
        dAF_dt    = -r2
        dE_GDH_dt = -r3
        dE_FDH_dt = -r4
        #dPPX_dt = r7

        return [dPakt_dt, dPslow_dt, dPinert_dt, dPPT_dt, dNADH_dt, dNAD_dt, dAF_dt, dE_GDH_dt, dE_FDH_dt]

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        v_PPO0, v_NAD0, v_GluDH, v_FDH = x

        #Substrate/Enzyme concentrations from starting volume
        c_PPO0 = v_PPO0 * p["c_PPOS"]
        c_PPO_inert = kinetic_params["f_inert"] * c_PPO0
        c_PPO_usable = c_PPO0 - c_PPO_inert
        P_akt0 = kinetic_params["f_Pakt0"] * c_PPO_usable
        P_slow0 = c_PPO_usable - P_akt0
        c_PPT0 = 0.0
        c_AF0 = c_PPO0
        c_NADH0 = 0.0
        c_NAD0 = v_NAD0 * p["c_NADS"]
        GluDH = v_GluDH * p["c_GluDHS"]
        FDH  = v_FDH * p["c_FDHS"]

        y0 = [P_akt0, P_slow0, c_PPO_inert, c_PPT0, c_NADH0, c_NAD0, c_AF0, GluDH, FDH]

        #Punishment if total volume is unrealistic
        if (v_PPO0+(p["c_PPOS"]/p["c_AFS"])*v_PPO0+v_NAD0+v_GluDH+v_FDH) > 1.0:
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        #event: target conversion is reached
        def event_X99(t, y):
            ppo = y[0] + y[1]
            return ppo - c_PPO_usable * (1 - p["X_PPO_target"])
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

        #solving for reaction time and PPO and PPT concentrations concentrations when X_target is reached
        hit = sol.t_events[0].size > 0
        tf  = float(sol.t_events[0][0]) if hit else sol.t[-1]
        y_f = sol.sol(tf) if hit else sol.y[:, -1]     #objective values, when target x is reached
        PPO = y_f[0] + y_f[1]
        PPT = y_f[3]

        #final PPO concentration after T_max
        PPO_final = sol.y[0, -1] + sol.y[1, -1]

        self.ppo_final_history.append(PPO)

        #objective functions
        sty     = PPT / (tf / 60)
        ton_e   = PPT / (GluDH + FDH)
        ton_cof = PPT / (c_NADH0 + c_NAD0)

        #for constraint
        X_PPO = 1.0 - PPO_final / c_PPO_usable

        return np.array([sty, ton_e, ton_cof, X_PPO])
