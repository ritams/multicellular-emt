"""
Plot mZEB vs SNAIL steady states to show three-state tristability
in the EMT core circuit (miR200/ZEB, miR34/SNAIL coupled circuits).
Uses Numba JIT compilation for fast ODE integration.
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numba import njit

# ── Precompute binomial coefficients ────────────────────────────────
C6 = np.array([1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0])  # C(6,i)
C2 = np.array([1.0, 2.0, 1.0])                            # C(2,i)

L_arr = np.array([1.0, 0.6, 0.3, 0.1, 0.05, 0.05, 0.05])
gamma_mRNA = np.array([0.0, 0.04, 0.2, 1.0, 1.0, 1.0, 1.0])
gamma_miRNA = np.array([0.0, 0.005, 0.05, 0.5, 0.5, 0.5, 0.5])


@njit
def hill_shifted(val, threshold, n, leakage):
    H = 1.0 / (1.0 + (val / threshold)**n)
    return H + leakage * (1.0 - H)


@njit
def ode_system(x, t):
    # Parameters
    g_miR34 = 1.35e3;   g_mSNAIL = 90.0;    g_SNAIL = 0.1e3
    g_miR200 = 2.1e3;   g_mZEB = 11.0;      g_ZEB = 0.1e3

    k_miR34 = 0.05;     k_mSNAIL = 0.5;     k_SNAIL = 0.125
    k_miR200 = 0.05;    k_mZEB = 0.5;       k_ZEB = 0.1

    t_miR34_SNAIL = 300e3;  t_mSNAIL_SNAIL = 200e3
    t_miR34_ZEB = 600e3;    t_miR34 = 10e3;     t_mSNAIL_I = 50e3
    t_miR200_ZEB = 220e3;   t_miR200_SNAIL = 180e3
    t_mZEB_ZEB = 25e3;      t_mZEB_SNAIL = 180e3;   t_miR200 = 10e3

    n_miR34_SNAIL = 1;  n_miR34_ZEB = 1;    n_mSNAIL_SNAIL = 1
    n_mSNAIL_I = 1;     n_miR200_ZEB = 3;   n_miR200_SNAIL = 2
    n_mZEB_ZEB = 2;     n_mZEB_SNAIL = 2

    l_miR34_SNAIL = 0.1;   l_mSNAIL_SNAIL = 0.1
    l_miR34_ZEB = 0.2;     l_mSNAIL_I = 10.0
    l_miR200_ZEB = 0.1;    l_miR200_SNAIL = 0.1
    l_mZEB_ZEB = 7.5;      l_mZEB_SNAIL = 10.0

    # x = [miR200, mZEB, ZEB, SNAIL, mSNAIL, miR34, I]

    # miR200-mZEB binding (6 sites)
    degrad_miR200 = 0.0; degrad_mZEB = 0.0; trans_mZEB = 0.0
    ratio_miR200 = x[0] / t_miR200
    denom6 = (1.0 + ratio_miR200)**6
    for i in range(7):
        fac = ratio_miR200**i / denom6
        degrad_miR200 += gamma_miRNA[i] * C6[i] * i * fac
        degrad_mZEB   += gamma_mRNA[i]  * C6[i] * fac
        trans_mZEB    += L_arr[i]        * C6[i] * fac

    # miR34-mSNAIL binding (2 sites)
    degrad_miR34 = 0.0; degrad_mSNAIL = 0.0; trans_mSNAIL = 0.0
    ratio_miR34 = x[5] / t_miR34
    denom2 = (1.0 + ratio_miR34)**2
    for i in range(3):
        fac = ratio_miR34**i / denom2
        degrad_miR34  += gamma_miRNA[i] * C2[i] * i * fac
        degrad_mSNAIL += gamma_mRNA[i]  * C2[i] * fac
        trans_mSNAIL  += L_arr[i]       * C2[i] * fac

    # Hill functions
    H_miR200_ZEB   = hill_shifted(x[2], t_miR200_ZEB,   n_miR200_ZEB,   l_miR200_ZEB)
    H_miR200_SNAIL = hill_shifted(x[3], t_miR200_SNAIL, n_miR200_SNAIL, l_miR200_SNAIL)
    H_mZEB_ZEB     = hill_shifted(x[2], t_mZEB_ZEB,     n_mZEB_ZEB,     l_mZEB_ZEB)
    H_mZEB_SNAIL   = hill_shifted(x[3], t_mZEB_SNAIL,   n_mZEB_SNAIL,   l_mZEB_SNAIL)
    H_miR34_SNAIL  = hill_shifted(x[3], t_miR34_SNAIL,  n_miR34_SNAIL,  l_miR34_SNAIL)
    H_miR34_ZEB    = hill_shifted(x[2], t_miR34_ZEB,    n_miR34_ZEB,    l_miR34_ZEB)
    H_mSNAIL_SNAIL = hill_shifted(x[3], t_mSNAIL_SNAIL, n_mSNAIL_SNAIL, l_mSNAIL_SNAIL)
    H_mSNAIL_I     = hill_shifted(x[6], t_mSNAIL_I,    n_mSNAIL_I,    l_mSNAIL_I)

    dxdt = np.zeros(7)
    dxdt[0] = g_miR200 * H_miR200_ZEB * H_miR200_SNAIL - x[1]*degrad_miR200 - k_miR200*x[0]
    dxdt[1] = g_mZEB * H_mZEB_ZEB * H_mZEB_SNAIL - x[1]*degrad_mZEB - k_mZEB*x[1]
    dxdt[2] = g_ZEB * x[1] * trans_mZEB - k_ZEB * x[2]
    dxdt[3] = g_SNAIL * x[4] * trans_mSNAIL - k_SNAIL * x[3]
    dxdt[4] = g_mSNAIL * H_mSNAIL_I * H_mSNAIL_SNAIL - x[4]*degrad_mSNAIL - k_mSNAIL*x[4]
    dxdt[5] = g_miR34 * H_miR34_ZEB * H_miR34_SNAIL - x[4]*degrad_miR34 - k_miR34*x[5]
    dxdt[6] = 0.0
    return dxdt


def sweep(I_values, x0):
    """Sweep I values sequentially, using previous steady state as next IC."""
    results = np.empty((len(I_values), 7))
    state = x0.copy()
    t_span = np.linspace(0, 7500, 500)
    for idx, I_val in enumerate(I_values):
        state[6] = I_val
        sol = odeint(ode_system, state, t_span)
        state = sol[-1].copy()
        results[idx] = state
    return results


# ── Warm up Numba JIT ──────────────────────────────────────────────
print("JIT compiling...")
_ = ode_system(np.zeros(7), 0.0)

# ── Run hysteresis sweeps ───────────────────────────────────────────
NUMPOINTS = 1000
I_forward  = np.linspace(20e3, 120e3, NUMPOINTS)
I_backward = np.linspace(120e3, 20e3, NUMPOINTS)
I_mid_back = np.linspace(65e3, 20e3, NUMPOINTS)

x0_low = np.zeros(7)

print("Running forward sweep (E → M)...")
res_fwd = sweep(I_forward, x0_low)

print("Running backward sweep (M → E)...")
res_bwd = sweep(I_backward, res_fwd[-1])

print("Running mid-backward sweep (hybrid branch)...")
x0_mid = np.zeros(7)
x0_mid[6] = 65e3
t_init = np.linspace(0, 7500, 500)
sol_mid = odeint(ode_system, x0_mid, t_init)
res_mid = sweep(I_mid_back, sol_mid[-1])

# ── Extract SNAIL (x[3]) and mZEB (x[1]) ───────────────────────────
snail_fwd = res_fwd[:, 3];  mzeb_fwd = res_fwd[:, 1]
snail_bwd = res_bwd[:, 3];  mzeb_bwd = res_bwd[:, 1]
snail_mid = res_mid[:, 3];  mzeb_mid = res_mid[:, 1]

# ── Plot ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax1 = axes[0]
ax1.plot(I_forward / 1e3, mzeb_fwd, 'b.', markersize=1, label='Forward (E→M)')
ax1.plot(I_backward / 1e3, mzeb_bwd, 'r.', markersize=1, label='Backward (M→E)')
ax1.plot(I_mid_back / 1e3, mzeb_mid, 'g.', markersize=1, label='Mid-backward (hybrid)')
ax1.set_xlabel('Input Signal I (×10³)', fontsize=12)
ax1.set_ylabel('mZEB (steady state)', fontsize=12)
ax1.set_title('Bifurcation Diagram', fontsize=13)
ax1.legend(fontsize=9, markerscale=8)

ax2 = axes[1]
ax2.plot(snail_fwd, mzeb_fwd, 'b.', markersize=1.5, label='Forward (E→M)')
ax2.plot(snail_bwd, mzeb_bwd, 'r.', markersize=1.5, label='Backward (M→E)')
ax2.plot(snail_mid, mzeb_mid, 'g.', markersize=1.5, label='Mid-backward (hybrid)')
ax2.set_xlabel('SNAIL (protein, steady state)', fontsize=12)
ax2.set_ylabel('mZEB (steady state)', fontsize=12)
ax2.set_title('mZEB vs SNAIL — Three-State Tristability', fontsize=13)
ax2.legend(fontsize=9, markerscale=8)

plt.tight_layout()
plt.savefig('tristability_mZEB_vs_SNAIL.png', dpi=200, bbox_inches='tight')
plt.show()
print("Saved: tristability_mZEB_vs_SNAIL.png")
