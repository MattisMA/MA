import numpy as np
from scipy.integrate import solve_ivp
from enzyme_kinetics import kinetics


class BatchReactor:
    """Batch reactor simulation: material balances + reaction simulation."""

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    2000,
            "c_AFS":     20000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  180,
            "c_FDHS":  70.5,

            "X_PPO_target": 0.999,         #target conversion
        }
        self.ppo_final_history = []

    #Material balances==============================================================================================================================================================
    def balances(self, t, y):
        r1, r2, r3, r4, r5, r6 = kinetics(y)

        #Material balances
        dPPO_dt   = -r1 #- r7
        dPPT_dt   = +r1
        dNADH_dt  = -r1 + r2 - r6
        dNAD_dt   = +r1 - r2 - r5
        dAF_dt    = -r2
        dE_GDH_dt = -r3
        dE_FDH_dt = -r4
        #dPPX_dt = r7

        return [dPPO_dt, dPPT_dt, dNADH_dt, dNAD_dt, dAF_dt, dE_GDH_dt, dE_FDH_dt]

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        v_PPO0, v_NAD0, v_GluDH, v_FDH = x

        #Substrate/Enzyme concentrations from starting volume
        c_PPO0 = v_PPO0 * p["c_PPOS"]
        c_PPT0 = 0.0
        c_AF0 = c_PPO0
        c_NADH0 = 0.0
        c_NAD0 = v_NAD0 * p["c_NADS"]
        GluDH = v_GluDH * p["c_GluDHS"]
        FDH  = v_FDH * p["c_FDHS"]

        y0 = [c_PPO0, c_PPT0, c_NADH0, c_NAD0, c_AF0, GluDH, FDH]

        #Punishment if total volume is unrealistic
        if (v_PPO0+(p["c_PPOS"]/p["c_AFS"])*v_PPO0+v_NAD0+v_GluDH+v_FDH) > 1.0:
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        #event: target conversion is reached
        def event_X99(t, y):
            return y[0] - y0[0] * (1 - p["X_PPO_target"])
        event_X99.terminal = True
        event_X99.direction = -1

        sol = solve_ivp(
            fun=self.balances,
            t_span=(0.0, 1e6),
            y0=y0,
            method="BDF",
            events=event_X99,
            rtol=1e-8,
            atol=1e-10,
        )

        #solving for reaction time and final PPO and PPT concentrations
        tf = sol.t[-1]
        PPO = sol.y[0, -1]
        PPT = sol.y[1, -1]

        self.ppo_final_history.append(PPO)

        #objective functions
        sty     = PPT / (tf / 60)
        ton_e   = PPT / (GluDH + FDH)
        ton_cof = PPT / (c_NADH0 + c_NAD0)

        #for constraint
        X_PPO = 1.0 - PPO / c_PPO0

        return np.array([sty, ton_e, ton_cof, X_PPO])
