import numpy as np

#Kinetic parameters==============================================================================================================================================================
kinetic_params = {
    #FDH kinetics
    "v_max1": 3.42,
    "Km_For": 8.70,
    "Km_NAD": 0.039,
    "Ki_NADH": 0.030,
    "Kis_TMP": 740.0,

    #LeuDH kinetics
    "v_max2": 19.10,
    "Km_TMP": 21.98,
    "Km_TLeu": 24.61,
    "Ki_TMP": 1838.0,
    "Km_NADH": 0.022,

    "T_r":  25.0,           #reaction temperature
    "kd_NAD": 0.00032268,   #deactivation rate for cofactor
}

#Reaction rates r1-r6==============================================================================================================================================================
def kinetics(y, p=kinetic_params):
    TMP, TLeu, NADH, NAD, For, E_LeuDH, E_FDH = np.maximum(y, 0.0)

    #GluDH
    r1 = E_LeuDH * TMP/(p["Km_TMP"] * (1 + TLeu/p["Ki_TLeu"]) + TMP **2/p["Ki_TMP"]) * NADH/(p["Km_NADH"] + NADH)

    #FDH
    r2 = E_FDH * For * NAD / ((1+TMP/p["Kis_TMP"])*(p["Km_For"]+For)*(p["Km_NAD"]*(1+NADH/p["Ki_NADH"])+NAD))

    if NAD <= 0 and r2 > 0:
        r2 = 0.0

    #Enzyme deactivation
    kd_E = 3.4761e-6 * np.exp(0.21017 * p["T_r"]) / 60
    r3 = kd_E * E_LeuDH
    r4 = kd_E * E_FDH

    #Cofactor deactivation
    r5 = p["kd_NAD"] * NAD
    r6 = p["kd_NAD"] * NADH

    return r1, r2, r3, r4, r5, r6