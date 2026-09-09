import numpy as np
from scipy.integrate import solve_ivp
from enzyme_kinetics_GluDH import kinetics, kinetic_params

"""Fed-batch reactor with constant feed rate starting at a given time"""

#state vector y = [P_akt, P_slow, I_inert, PPT, NADH, NAD, AF, E_GDH, E_FDH, V/V0, n_PPO_fed]
class Reactor:
    v_names = ["PPO [mL/mL]", "AF [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]",
                 "f_PPO [1/min]", "t_feed_start [min]"]
    LOWER = np.array([0.0001, 0.0001, 0.0005, 0.0001, 0.0001, 1e-8, 1.0])
    UPPER = np.array([0.95, 0.2, 0.05, 0.20, 0.20, 0.1, 1200.0])
    weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])   #only the initial charge occupies V0

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    1000,
            "c_AFS":     8000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  120,
            "c_FDHS":  100,

            "X_target": 0.999,         #target conversion
            "V_max":        1.5,           #maximum volume [V/V0]
            "t_setup":      0.0,           #setup time per batch [h]
            "t_max":        1200,          #maximum Batch time total [min]
        }

        #speciation of the PPO stock solution, same split as the initial charge
        kp = kinetic_params
        c_in_inert = kp["f_inert"] * self.params["c_PPOS"]
        c_in_usable = self.params["c_PPOS"] - c_in_inert
        c_in_Pakt = kp["f_Pakt0"] * c_in_usable
        self.feed_spec = (c_in_Pakt,                 #P_akt
                          c_in_usable - c_in_Pakt,   #P_slow
                          c_in_inert,                #I_inert
                          c_in_usable)               #usable, needed for the conversion
        self.ppo_final_history = []

    #Material balances==============================================================================================================================================================
    def balances(self, t, y, feeds, feed_on=1):
        p = self.params
        r1, r2, r3, r4, r5, r6, r7 = kinetics(y[:9])
        P_akt, P_slow, I_inert, PPT, NADH, NAD, AF, E_GDH, E_FDH = y[:9]
        V = y[9]

        f_PPO, f_AF = feed_on * np.asarray(feeds, dtype=float)
        f_total = f_PPO + f_AF
        c_in_Pakt, c_in_Pslow, c_in_inert, c_in_usable = self.feed_spec

        #Material balances
        dPakt_dt   = f_PPO * c_in_Pakt  - f_total * P_akt   - r1 - r7
        dPslow_dt  = f_PPO * c_in_Pslow - f_total * P_slow        + r7
        dIinert_dt = f_PPO * c_in_inert - f_total * I_inert
        dPPT_dt    = -f_total * PPT   + r1
        dNADH_dt   = -f_total * NADH - r1 + r2 - r6
        dNAD_dt    = -f_total * NAD + r1 - r2 - r5
        dAF_dt     = f_AF * p["c_AFS"] - f_total * AF - r2
        dE_GDH_dt  = -f_total * E_GDH - r3
        dE_FDH_dt  = -f_total * E_FDH - r4
        dV_dt      = f_total * V

        #fed usable PPO for conversion
        dn_PPO_dt = f_PPO * c_in_usable * V

        return [dPakt_dt, dPslow_dt, dIinert_dt, dPPT_dt, dNADH_dt, dNAD_dt, dAF_dt,
                dE_GDH_dt, dE_FDH_dt, dV_dt, dn_PPO_dt]

    #Phases: batch till t_feed_start -> feed till V_max -> batch till t_max===============================================
    def run(self, y0, feeds, t_feed_start):
        p = self.params
        y0 = np.asarray(y0, dtype=float)
        n_PPO0 = y0[0] + y0[1]                          #usable PPO charged
        t_end = p["t_max"]

        #event: target conversion reached sol.t_events[0]
        def event_X99(t, y):
            return (y[0] + y[1]) * y[9] - (1 - p["X_target"]) * (n_PPO0 + y[10])
        event_X99.terminal = False
        event_X99.direction = -1

        #event: maximum volume reached
        def event_V_max(t, y):
            return y[9] - p["V_max"]
        event_V_max.terminal = True
        event_V_max.direction = 1

        #Phase 1: batch until the feed starts
        sol1 = solve_ivp(
            fun=lambda t, y: self.balances(t, y, feeds, feed_on=0),
            t_span=(0.0, min(t_feed_start, t_end)),
            y0=y0,
            method="BDF",
            events=[event_X99],
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )
        sols = [sol1]
        if sol1.status == -1 or sol1.t[-1] >= t_end:
            return sols

        #Phase 2: feed until the volume limit is reached
        sol2 = solve_ivp(
            fun=lambda t, y: self.balances(t, y, feeds, feed_on=1),
            t_span=(sol1.t[-1], t_end),
            y0=sol1.y[:, -1],
            method="BDF",
            events=[event_X99,event_V_max],
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )
        sols.append(sol2)
        if sol2.status == -1 or sol2.t[-1] >= t_end:
            return sols

        #Phase 3: batch again until the maximum time
        sol3 = solve_ivp(
            fun=lambda t, y: self.balances(t, y, feeds, feed_on=0),
            t_span=(sol2.t[-1], t_end),
            y0=sol2.y[:, -1],
            method="BDF",
            events=[event_X99],
            rtol=1e-8,
            atol=1e-10,
            dense_output=True,
        )
        sols.append(sol3)

        return sols

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x):
        p = self.params
        v_PPO0, v_AF0, v_NAD0, v_GluDH, v_FDH, f_PPO, t_feed_start = x

        #AF feed rate from v_PPO0 and f_PPO
        V_feed = p["V_max"] - 1.0
        AF_missing = (p["c_PPOS"] / p["c_AFS"]) * (v_PPO0 + V_feed) - v_AF0
        AF_share = v_AF0 + V_feed - (p["c_PPOS"] / p["c_AFS"]) * v_PPO0
    
        f_AF = f_PPO * AF_missing / AF_share

        #staring concentrations, PPO split into active/slow/inert
        c_PPO0 = v_PPO0 * p["c_PPOS"]
        I_inert0 = kinetic_params["f_inert"] * c_PPO0
        c_PPO_usable = c_PPO0 - I_inert0
        P_akt0 = kinetic_params["f_Pakt0"] * c_PPO_usable
        P_slow0 = c_PPO_usable - P_akt0
        c_PPT0 = 0.0
        c_AF0 = v_AF0 * p["c_AFS"]
        c_NADH0 = 0.0
        c_NAD0 = v_NAD0 * p["c_NADS"]
        GluDH = v_GluDH * p["c_GluDHS"]
        FDH = v_FDH * p["c_FDHS"]

        y0 = [P_akt0, P_slow0, I_inert0, c_PPT0, c_NADH0, c_NAD0, c_AF0, GluDH, FDH, 1.0, 0.0]
        feeds = np.array([f_PPO, f_AF])

        try:
            sols = self.run(y0, feeds, t_feed_start)
        except Exception:
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        if any(sol.status == -1 for sol in sols):
            self.ppo_final_history.append(np.nan)
            return np.array([0, 0, 0, 0])

        #state at the first time the target conversion is reached, event_X99 is always event 0
        tf, y_event = None, None
        for sol in sols:
            if sol.t_events[0].size > 0:
                tf, y_event = float(sol.t_events[0][0]), sol.y_events[0][0]
                break

        #state at the end of the last phase, used for the conversion constraint
        y_end = sols[-1].y[:, -1]

        #target not reached, the run is judged by its state at the end
        if y_event is None:
            tf, y_event = float(sols[-1].t[-1]), y_end

        V = y_event[9]
        PPO, PPT = y_event[0] + y_event[1], y_event[3]

        n_PPT = PPT * V                                 #product per L initial volume
        n_cof = c_NADH0 + c_NAD0

        self.ppo_final_history.append(PPO * V)

        #objective functions, evaluated when the target conversion is reached
        sty     = PPT / (tf / 60 + p["t_setup"])
        ton_e   = n_PPT / (GluDH + FDH)
        ton_cof = n_PPT / n_cof

        #for constraint, evaluated at the end of the horizon: everything fed has to be converted
        n_PPO_in = c_PPO_usable + y_end[10]             #charged + fed usable PPO
        X_PPO = 1.0 - (y_end[0] + y_end[1]) * y_end[9] / n_PPO_in

        return np.array([sty, ton_e, ton_cof, X_PPO])
