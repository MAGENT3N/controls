"""
LQR Control of a Spring-Mass-Damper System
==========================================
System:  m*x'' + c*x' + k*x = u
States:  x1 = position, x2 = velocity
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ── System Parameters ────────────────────────────────────────────────────────
m = 1.0   # kg
c = 0.5   # N·s/m
k = 2.0   # N/m

A = np.array([[0,    1  ],
              [-k/m, -c/m]])
B = np.array([[0  ],
              [1/m]])

# ── LQR Design ───────────────────────────────────────────────────────────────
Q = np.diag([10.0, 1.0])   # penalise position (x1) more than velocity (x2)
R = np.array([[1.0]])       # control effort weight

P = solve_continuous_are(A, B, Q, R)          # solve CARE: ATP + PA - PBR⁻¹BᵀP + Q = 0
K = np.linalg.inv(R) @ B.T @ P               # optimal gain:  K = R⁻¹BᵀP

print(f"Gain matrix K = {K}")
print(f"Closed-loop poles: {np.linalg.eigvals(A - B @ K)}")

# ── Simulation Settings ──────────────────────────────────────────────────────
x0     = [1.0, 0.0]                           # initial condition: x(0) = [1, 0]ᵀ
t_span = (0, 20)                              # simulation horizon [s]
t_eval = np.linspace(0, 20, 2000)

# ── Open-Loop (u = 0) ────────────────────────────────────────────────────────
def ode_open(t, x):
    return A @ x

sol_open = solve_ivp(ode_open, t_span, x0, method='RK45',
                     t_eval=t_eval, rtol=1e-8)

# ── Closed-Loop (u = -Kx) ────────────────────────────────────────────────────
def ode_closed(t, x):
    u = -(K @ x)
    return (A @ x) + (B @ u).flatten()

sol_lqr = solve_ivp(ode_closed, t_span, x0, method='RK45',
                    t_eval=t_eval, rtol=1e-8)

u_lqr = np.array([-(K @ sol_lqr.y[:, i]).item()
                  for i in range(len(t_eval))])

# ── Performance Metrics ──────────────────────────────────────────────────────
threshold   = 0.02 * abs(x0[0])
settled_idx = np.where(np.abs(sol_lqr.y[0]) < threshold)[0]
settling_time = sol_lqr.t[settled_idx[0]] if len(settled_idx) else float('inf')
neg_vals    = sol_lqr.y[0][sol_lqr.y[0] < 0]
overshoot   = float(np.max(np.abs(neg_vals)) / abs(x0[0]) * 100) if len(neg_vals) else 0.0

print(f"\nSettling time (2%): {settling_time:.2f} s")
print(f"Overshoot:          {overshoot:.1f} %")
print(f"Peak control effort:{np.max(np.abs(u_lqr)):.2f} N")

# ── Figure 1 — Open-Loop Response ────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 4))
ax.plot(sol_open.t, sol_open.y[0], 'b-',  lw=1.8, label=r'$x_1(t)$ — Position (m)')
ax.plot(sol_open.t, sol_open.y[1], 'r--', lw=1.8, label=r'$x_2(t)$ — Velocity (m/s)')
ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_xlabel('Time (s)');  ax.set_ylabel('State')
ax.set_title('Fig. 1 — Uncontrolled (Open-Loop) Response')
ax.legend();  ax.grid(alpha=0.3);  ax.set_xlim([0, 20])
fig1.tight_layout()
fig1.savefig('fig1_openloop.png', dpi=150)

# ── Figure 2 — LQR Controlled Response ───────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax1.plot(sol_lqr.t, sol_lqr.y[0], 'b-',  lw=1.8, label=r'$x_1(t)$ — Position (m)')
ax1.plot(sol_lqr.t, sol_lqr.y[1], 'r--', lw=1.8, label=r'$x_2(t)$ — Velocity (m/s)')
ax1.axhline(0, color='k', lw=0.5, ls=':')
ax1.set_ylabel('State');  ax1.legend();  ax1.grid(alpha=0.3)
ax1.set_title('Fig. 2 — LQR Controlled Response')

ax2.plot(sol_lqr.t, u_lqr, 'g-', lw=1.8, label=r'$u(t)$ — Control Force (N)')
ax2.axhline(0, color='k', lw=0.5, ls=':')
ax2.set_xlabel('Time (s)');  ax2.set_ylabel('Control Force (N)')
ax2.legend();  ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig('fig2_lqr.png', dpi=150)

# ── Figure 3 — Q/R Parametric Study ─────────────────────────────────────────
configs = [
    ('Q=diag(1,1),  R=1',  np.diag([1.0,  1.0]), np.array([[1.0]]),  'C0', '-'),
    ('Q=diag(10,1), R=1',  np.diag([10.0, 1.0]), np.array([[1.0]]),  'C1', '--'),
    ('Q=diag(50,1), R=1',  np.diag([50.0, 1.0]), np.array([[1.0]]),  'C2', '-.'),
    ('Q=diag(10,1), R=10', np.diag([10.0, 1.0]), np.array([[10.0]]), 'C3', ':'),
]

fig3, ax = plt.subplots(figsize=(8, 4.5))
for label, Qi, Ri, color, ls in configs:
    Pi = solve_continuous_are(A, B, Qi, Ri)
    Ki = np.linalg.inv(Ri) @ B.T @ Pi
    sol_i = solve_ivp(lambda t, x, Ki=Ki: (A - B @ Ki) @ x,
                      t_span, x0, method='RK45', t_eval=t_eval, rtol=1e-8)
    ax.plot(sol_i.t, sol_i.y[0], color=color, ls=ls, lw=1.8, label=label)

ax.axhline(0, color='k', lw=0.5, ls=':')
ax.set_xlabel('Time (s)');  ax.set_ylabel(r'$x_1(t)$ — Position (m)')
ax.set_title('Fig. 3 — Effect of Q and R on LQR Response')
ax.legend(fontsize=9);  ax.grid(alpha=0.3);  ax.set_xlim([0, 15])
fig3.tight_layout()
fig3.savefig('fig3_qr_variation.png', dpi=150)

plt.show()
print("\nDone. Figures saved: fig1_openloop.png, fig2_lqr.png, fig3_qr_variation.png")
