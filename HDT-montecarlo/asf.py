# --- Corredera–manivela con descentre (offset) ---
# Parámetros del mecanismo
L2 = 40.0   # mm (manivela)
L3 = 100.0  # mm (biela)
e  = 15.0   # mm (descentre)
v_slider = 10_000.0  # mm/s (10 m/s)

import numpy as np
import math
import matplotlib.pyplot as plt

# ========= 1) Geometría (lazo vectorial) =========
# Ecuaciones:
#   L2*cosθ2 + L3*cosθ3 = s
#   L2*sinθ2 + L3*sinθ3 = e
# Eliminando θ3:
#   (s - L2 cosθ2)^2 + (e - L2 sinθ2)^2 = L3^2
#   => A cosθ2 + B sinθ2 = C
#   A = 2 s L2, B = 2 e L2, C = s^2 + e^2 + L2^2 - L3^2
#   θ2 = φ ± arccos(C/R),   R = sqrt(A^2 + B^2),   φ = atan2(B, A)

def s_plus_from_theta2(theta2):
    """Rama física del slider: s = xB + sqrt(L3^2 - (e - yB)^2)"""
    xB = L2*np.cos(theta2)
    yB = L2*np.sin(theta2)
    rad = L3**2 - (e - yB)**2
    rad = np.maximum(rad, 0.0)
    return xB + np.sqrt(rad)

def stroke_limits_plus(n=20001):
    """s_min, s_max usando SOLO la rama + (la física)."""
    th = np.linspace(0, 2*np.pi, n)
    s_vals = s_plus_from_theta2(th)
    return float(np.min(s_vals)), float(np.max(s_vals))

def theta2_from_s_plus(s):
    """Invierte s -> θ2 quedándose con la solución compatible con la rama +.
       Usa la fórmula cerrada y escoge entre las dos raíces la que reproduce s_plus."""
    A = 2*s*L2
    B = 2*e*L2
    C = s**2 + e**2 + L2**2 - L3**2
    R = math.hypot(A, B)
    # proteger por redondeos
    u = (C / R) if R > 0 else 1.0
    u = max(min(u, 1.0), -1.0)
    phi = math.atan2(B, A)

    # Dos candidatas
    th_a = (phi + math.acos(u)) % (2*np.pi)
    th_b = (phi - math.acos(u)) % (2*np.pi)

    # Evaluar cuál reproduce la rama +
    s_a = s_plus_from_theta2(th_a)
    s_b = s_plus_from_theta2(th_b)

    return th_a if abs(s_a - s) <= abs(s_b - s) else th_b

def theta3_from_theta2_s(theta2, s):
    """θ3 desde el lazo de posición."""
    c3 = (s - L2*math.cos(theta2)) / L3
    s3 = (e - L2*math.sin(theta2)) / L3
    # limitar por redondeo numérico
    c3 = max(min(c3, 1.0), -1.0)
    s3 = max(min(s3, 1.0), -1.0)
    return math.atan2(s3, c3)

# ========= 2) Velocidades (CIV: lazo derivado) =========
# Derivando el lazo:
#  -L2 sinθ2 * ω2 - L3 sinθ3 * ω3 = v_slider
#   L2 cosθ2 * ω2 + L3 cosθ3 * ω3 = 0
# Sistema 2x2 => [ω2, ω3]^T

def omegas_from_angles(theta2, theta3, v_s):
    a11 = -L2*math.sin(theta2); a12 = -L3*math.sin(theta3)
    a21 =  L2*math.cos(theta2); a22 =  L3*math.cos(theta3)
    det = a11*a22 - a12*a21
    if abs(det) < 1e-12:
        return np.nan, np.nan
    inv11, inv12 =  a22/det, -a12/det
    inv21, inv22 = -a21/det,  a11/det
    w2 = inv11*v_s + inv12*0.0
    w3 = inv21*v_s + inv22*0.0
    return w2, w3

# ========= 3) Simulación: slider a velocidad constante (ida y vuelta) =========
def simulate(N=4000):
    s_min, s_max = stroke_limits_plus()
    stroke = s_max - s_min
    t_half = stroke / v_slider
    T = 2*t_half

    t = np.linspace(0.0, T, N)
    # trayectoria lineal del slider (rama + en todo momento)
    s = np.where(t <= t_half, s_min + v_slider*t, s_max - v_slider*(t - t_half))

    th2 = np.zeros_like(t)
    th3 = np.zeros_like(t)
    w2  = np.zeros_like(t)
    w3  = np.zeros_like(t)

    for i, si in enumerate(s):
        th2_i = theta2_from_s_plus(float(si))
        th3_i = theta3_from_theta2_s(th2_i, float(si))
        w2_i, w3_i = omegas_from_angles(th2_i, th3_i, v_slider)

        th2[i], th3[i], w2[i], w3[i] = th2_i, th3_i, w2_i, w3_i

    # Desenrollar para continuidad y pasar a grados
    th2_deg = np.unwrap(th2) * 180/np.pi
    th3_deg = np.unwrap(th3) * 180/np.pi

    return {
        "t": t, "s": s,
        "theta2_deg": th2_deg,
        "theta3_deg": th3_deg,
        "omega2": w2, "omega3": w3,
        "stroke": stroke, "t_half": t_half, "T": T
    }

# ========= 4) Ejecutar y graficar =========
out = simulate(N=4000)

print(f"Stroke (recorrido): {out['stroke']:.3f} mm")
print(f"Medio ciclo: {out['t_half']:.6f} s")
print(f"Ciclo completo: {out['T']:.6f} s")

plt.figure()
plt.plot(out["t"], out["theta2_deg"])
plt.xlabel("Tiempo (s)")
plt.ylabel(r"$\theta_2$ (deg)")
plt.title(r"$\theta_2$ vs tiempo (continuo)")
plt.grid(True); plt.show()

plt.figure()
plt.plot(out["t"], out["theta3_deg"])
plt.xlabel("Tiempo (s)")
plt.ylabel(r"$\theta_3$ (deg)")
plt.title(r"$\theta_3$ vs tiempo (continuo)")
plt.grid(True); plt.show()

# (Opcional) Descomenta para ver velocidades por CIV:
# plt.figure(); plt.plot(out["t"], out["omega2"]); plt.xlabel("t (s)"); plt.ylabel(r"$\omega_2$ (rad/s)"); plt.grid(True); plt.show()
# plt.figure(); plt.plot(out["t"], out["omega3"]); plt.xlabel("t (s)"); plt.ylabel(r"$\omega_3$ (rad/s)"); plt.grid(True); plt.show()
