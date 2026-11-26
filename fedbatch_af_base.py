import numpy as np
from scipy.integrate import solve_ivp
from enzyme_kinetics_GluDH import kinetics, kinetic_params

#fedbatch base reactor: fed phase + batch phase, only af feed

#state vector y = [P_akt, P_slow, I_inert, PPT, NADH, NAD, AF, E_GDH, E_FDH, V/V0, n_AF_fed]
class FedBatchAFBase:
    #initial volumes [mL/mL V0]
    _v_names_charge = ["PPO [mL/mL]", "NAD [mL/mL]", "E_GDH [mL/mL]", "E_FDH [mL/mL]"]
    _LOWER_charge = [0.0001, 0.0005, 0.0001, 0.0001]
    _UPPER_charge = [0.95,   0.05,   0.20,   0.20]
    n_charge = len(_v_names_charge)

    #feed variables, set by the subclass: [f0, (shape parameters)]
    feed_names = []
    feed_LOWER = []
    feed_UPPER = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.v_names = cls._v_names_charge + list(cls.feed_names)
        cls.LOWER = np.array(cls._LOWER_charge + list(cls.feed_LOWER), dtype=float)
        cls.UPPER = np.array(cls._UPPER_charge + list(cls.feed_UPPER), dtype=float)
        cls.weights = np.array([1.0] * cls.n_charge + [0.0] * len(cls.feed_names))   #only the initial charge occupies V0

    def __init__(self):
        self.params = {
            #Stock solutions [mM|U/mL]
            "c_PPOS":    1000,
            "c_AFS":     8000,
            "c_NADHS":   20,
            "c_NADS":    20,
            "c_GluDHS":  100,
            "c_FDHS":    100,

            "X_PPO_target": 0.999,         #target conversion
            "AF_PPO_ratio": 1.0,           #feed stops at n_AF(fed) = ratio * n_PPO(charged)
            "V_max":        1.5,           #maximum volume [V/V0]
            "t_setup":      0.0,           #setup time per batch [h]
            "t_max":        1200,          #maximum Batch time total [min]
        }
        self.ppo_final_history = []
        

    #feed function, implemented by the subclass ======================================================================
    def feed_rate(self, t, feed_par):
        raise NotImplementedError
    
    def feed_coeffs(self, A, feed_par):
        raise NotImplementedError

    def feed_coeffs_from_x(self, x):
        p = self.params
        #required AF volume [mL stock / mL V0]: n_AF_target / c_AFS, n_PPO_total0 = v_PPO0 * c_PPOS
        A = p["AF_PPO_ratio"] * x[0] * p["c_PPOS"] / p["c_AFS"]
        return self.feed_coeffs(A, x[self.n_charge:])

    #Material balances==============================================================================================================================================================
    def balances(self, t, y, coeffs=None, feed_on=0):
        p = self.params
        r1, r2, r3, r4, r5, r6, r7 = kinetics(y[:9])
        P_akt, P_slow, I_inert, PPT, NADH, NAD, AF, E_GDH, E_FDH = y[:9]
        V = y[9]

        #f_AF = F_AF/V0 [1/min]: volumetric pump flow related to the INITIAL volume
        #-> constant f_AF means constant pump flow
        f_AF = self.feed_rate(t, coeffs) if feed_on else 0.0
        D = f_AF / V                         #dilution rate F_AF/V [1/min], V = V/V0

        #Material balances, only AF is fed -> every other species is only diluted
        dPakt_dt   = -D * P_akt   - r1 - r7
        dPslow_dt  = -D * P_slow       + r7
        dIinert_dt = -D * I_inert
        dPPT_dt    = -D * PPT     + r1
        dNADH_dt   = -D * NADH    - r1 + r2 - r6
        dNAD_dt    = -D * NAD     + r1 - r2 - r5
        dAF_dt     = D * (p["c_AFS"] - AF) - r2
        dE_GDH_dt  = -D * E_GDH   - r3
        dE_FDH_dt  = -D * E_FDH   - r4
        dV_dt      = f_AF

        #fed AF [mmol per L initial volume]
        dn_AF_dt = f_AF * p["c_AFS"]

        return [dPakt_dt, dPslow_dt, dIinert_dt, dPPT_dt, dNADH_dt, dNAD_dt, dAF_dt,
                dE_GDH_dt, dE_FDH_dt, dV_dt, dn_AF_dt]

    #Phases: AF feed from t = 0 till n_AF = n_PPO (or V_max) -> batch till t_max================================================
    def run(self, y0, t_feed, coeffs):
        p = self.params
        y0 = np.asarray(y0, dtype=float)
        n_PPO_usable0 = y0[0] + y0[1]
        t_end = p["t_max"]
        ivp = dict(method="BDF", rtol=1e-8, atol=1e-10, dense_output=True)

        #event: target conversion reached, always event 0 (only recorded)
        def event_X99(t, y):
            return (y[0] + y[1]) * y[9] - (1 - p["X_PPO_target"]) * n_PPO_usable0
        event_X99.terminal = False
        event_X99.direction = -1

        #event: maximum volume reached -> feed stops (safety)
        def event_V_max(t, y):
            return y[9] - p["V_max"]
        event_V_max.terminal = True
        event_V_max.direction = 1

        #Phase 1: AF feed from the start until n_AF = n_PPO or V_max
        sol1 = solve_ivp(
            lambda t, y: self.balances(t, y, coeffs, feed_on=1),
            (0.0, min(t_feed, t_end)), y0, events=[event_X99, event_V_max], **ivp)
        sols = [sol1]
        if sol1.status == -1 or sol1.t[-1] >= t_end:
            return sols

        #Phase 2: batch until the maximum time
        sol2 = solve_ivp(lambda t, y: self.balances(t, y, feed_on=0),
                         (sol1.t[-1], t_end), sol1.y[:, -1], events=[event_X99], **ivp)
        sols.append(sol2)
        return sols

    #Starting concentrations=====================================================================================================================================
    def initial_state(self, x):
        p = self.params
        v_PPO0, v_NAD0, v_GluDH, v_FDH = x[:self.n_charge]
        c_PPO0 = v_PPO0 * p["c_PPOS"]
        I_inert0 = kinetic_params["f_inert"] * c_PPO0
        c_PPO_usable = c_PPO0 - I_inert0
        P_akt0 = kinetic_params["f_Pakt0"] * c_PPO_usable
        P_slow0 = c_PPO_usable - P_akt0
        return np.array([P_akt0, P_slow0, I_inert0, 0.0, 0.0,
                         v_NAD0 * p["c_NADS"], 0.0,
                         v_GluDH * p["c_GluDHS"], v_FDH * p["c_FDHS"], 1.0, 0.0])

    #Reaction simulation===========================================================================================================================================
    def simulate(self, x, return_sols=False):
        p = self.params
        x = np.asarray(x, dtype=float)
        t_feed = x[self.n_charge:]
        coeffs = self.feed_coeffs_from_x(x)
        fail = (np.array([0, 0, 0, 0]), None) if return_sols else np.array([0, 0, 0, 0])

        #Punishment if total volume is unrealistic
        if np.sum(x[:self.n_charge]) > 1.0:
            self.ppo_final_history.append(np.nan)
            return fail

        y0 = self.initial_state(x)
        c_PPO_usable = y0[0] + y0[1]
        GluDH, FDH, c_NAD0 = y0[7], y0[8], y0[5]

        try:
            sols = self.run(y0, t_feed, coeffs)
        except Exception:
            self.ppo_final_history.append(np.nan)
            return fail

        if any(sol.status == -1 for sol in sols):
            self.ppo_final_history.append(np.nan)
            return fail

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
        n_cof = c_NAD0                                  #only NAD charged, no cofactor feed

        self.ppo_final_history.append(PPO * V)

        #objective functions, evaluated when the target conversion is reached
        sty     = PPT / (tf / 60 + p["t_setup"])
        ton_e   = n_PPT / (GluDH + FDH)
        ton_cof = n_PPT / n_cof

        #for constraint, evaluated at the end of the horizon (PPO is not fed -> charged usable PPO)
        X_PPO = 1.0 - (y_end[0] + y_end[1]) * y_end[9] / c_PPO_usable

        result = np.array([sty, ton_e, ton_cof, X_PPO])
        return (result, sols) if return_sols else result
