import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from batch_reactor import BatchReactor

reactor = BatchReactor()
params = reactor.params

#Starting conditions=====================================================================================================================================================
v0 = np.array([
    0.703010653746265,    #PPO
    0.00010506818908977,  #NAD
    0.129978608582047,    #GDH
    0.0702339059014108,   #FDH
    #0.00,   # PPX
])

if np.sum(v0) > 1.0:
    raise ValueError(
        f"Summe der Volumenfraktionen = {np.sum(v0):.4f} > 1 — keine realistische Zusammensetzung"
    )

c_stock = np.array([
    params["c_PPOS"],
    params["c_NADHS"],
    params["c_NADS"],
    params["c_AFS"],
    params["c_GluDHS"],
    params["c_FDHS"],
])

c0 = np.array([
    v0[0] * c_stock[0],            #PPO
    0.0,                           #PPT
    0.0,                           #NADH
    v0[1] * c_stock[2],            #NAD
    v0[0] * c_stock[0]/c_stock[3], #AF
    v0[2] * c_stock[4],            #GDH
    v0[3] * c_stock[5],            #FDH
])

#Abbruch============================================================================================================
def event_X99(t, y):
    return y[0] - c0[0] * (1 - params["X_PPO_target"])
event_X99.terminal  = True
event_X99.direction = -1


#Solve============================================================================================================
sol = solve_ivp(
    fun=reactor.balances,
    t_span=(0.0, 1e6),
    y0=c0,
    method="BDF",
    events=event_X99,
    rtol=1e-8,
    atol=1e-10,
    dense_output=True,
)

t = np.linspace(0.0, sol.t[-1], 500)
PPO, PPT, NADH, NAD, AF, E_GDH, E_FDH = sol.sol(t)

#Objective functions============================================================================================================

t_h = t / 60
t_setup = 0
t_total = t_h + t_setup

STY = PPT / t_total
TON_e  =  PPT / (c0[5] + c0[6])
TON_c =  PPT / (c0[2] + c0[3]) 

print(f"  STY          =  {STY[-1]:.4f} mmol/L/h")
print(f"  TON_Enzym    =  {TON_e[-1]:.6f} mmol/U")
print(f"  TON_Cofaktor =  {TON_c[-1]:.6f} mmol/mmol")
print(f"  Umsatz PPO   =  {(1 - PPO[-1] / c0[0]) * 100:.2f} %")
print(f"  Verweilzeit  =  {sol.t[-1]:.2f} min  ({sol.t[-1]/60:.4f} h)")

#Plots============================================================================================================
fig, axes = plt.subplots(2,2, figsize=(13, 9))
fig.suptitle("Batch-Reaktor", fontsize=13)

#Substrat/Produkt
ax = axes[0, 0]
ax.plot(t, PPO, label="PPO", color="steelblue")
ax.plot(t, PPT, label="PPT", color="darkorange", linestyle="--")
ax.set_xlabel("Zeit [min]")
ax.set_ylabel("Konzentration [mmol/L]")
ax.set_title("PPO -> PPT")
ax.legend()
ax.grid(True, linestyle=":")

#Umsatz
ax = axes[0, 1]
ax.plot(t, 1 - PPO / c0[0], color="teal")
ax.axhline((1 - PPO[-1] / c0[0]), color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel("Zeit [min]")
ax.set_ylabel("Umsatz X  [–]")
ax.set_title("Umsatz PPO")
ax.set_ylim(0, 1.05)
ax.grid(True, linestyle=":")

#NAD/NADH
NAD_total = NADH + NAD
ax = axes[1, 0]
ax.plot(t, NADH,      label="NADH",          color="purple")
ax.plot(t, NAD,       label="NAD",          color="orchid", linestyle="--")
ax.plot(t, NAD_total, label="NAD (gesamt)",  color="gray",   linestyle=":")
ax.set_xlabel("Zeit [min]")
ax.set_ylabel("Konzentration [mmol/L]")
ax.set_title("NAD/NADH")
ax.legend()
ax.grid(True, linestyle=":")


plt.tight_layout()
plt.savefig("batch_reaktor_simulation2.png", dpi=150)
plt.show()
